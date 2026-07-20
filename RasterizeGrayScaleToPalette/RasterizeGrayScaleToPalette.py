import os
import subprocess
from sys import argv
import numpy as np
import imageio.v3 as iio
from PIL import Image, UnidentifiedImageError
from os.path import splitext, dirname
import time


def apply_palette(texconv, grayscale_path, palette_path, scale, out_path, max_size, dds_format):
	"""Apply a palette row to a grayscale image.

	grayscale_path: path to grayscale
	palette_path: path to palette (each row is a palette)
	scale: 0.0..1.0 selects row in palette
	out_path: output path
	"""
	# read images
	grayscale = load_dds_safe(grayscale_path, texconv)
	g = np.array(grayscale)
	palette = load_dds_safe(palette_path, texconv)
	p = np.array(palette)

	# ensure grayscale is single channel 0..255 while preserving alpha when present
	alpha = None
	if g.ndim == 2:
		g = g.astype(np.uint8)
	elif g.ndim == 3:
		if g.shape[2] == 2:
			alpha = g[..., 1].astype(np.uint8)
			g = g[..., 0]
		elif g.shape[2] == 4:
			alpha = g[..., 3].astype(np.uint8)
			g = g[..., 0]
		else:
			g = g[..., 0]
	g = g.astype(np.uint8)

	# palette should be H x W x C
	if p.ndim == 2:
		# single channel palette -> expand
		p = np.stack([p, p, p], axis=-1)
	if p.shape[2] == 4:
		palette_rgb = p[..., :3]
	else:
		palette_rgb = p

	ph = palette_rgb.shape[0]
	pw = palette_rgb.shape[1]

	# choose row from palette
	row = round(np.clip(scale, 0.0, 1.0) * (ph - 1))
	palette_row = palette_rgb[row]  # shape (width, 3)

	# if palette width != 256 and grayscale uses 0..255, scale indices
	# map grayscale values (0..255) to columns 0..pw-1
	indices = (g.astype(np.float32) / 255.0 * (pw - 1)).round().astype(int)

	# build colored image
	h, w = g.shape

	if alpha is not None:
		out = np.zeros((h, w, 4), dtype=np.uint8)
		out[..., 0] = palette_row[indices, 0]
		out[..., 1] = palette_row[indices, 1]
		out[..., 2] = palette_row[indices, 2]
		out[..., 3] = alpha
	else:
		out = np.zeros((h, w, 3), dtype=np.uint8)
		out[..., 0] = palette_row[indices, 0]
		out[..., 1] = palette_row[indices, 1]
		out[..., 2] = palette_row[indices, 2]

	size = max(h,w)
	while size > max_size:
		h = h // 2
		w = w // 2
		size = max(h,w)

	resized_image = Image.fromarray(out).resize((w, h), Image.LANCZOS)

	final_image = np.array(resized_image)

	save_texture(out_path, final_image, texconv, dds_format)


def reverse_palette(texconv, input_paths, grayscale_out_path, palette_out_path, max_size, dds_format, palette_limit=256):
	"""Build a shared grayscale image and palette rows for multiple input textures."""
	arrays = [load_texture_array(path, texconv) for path in input_paths]

	# Resize to a common size first so the derived grayscale image and palette fit each texture.
	target_height, target_width = arrays[0].shape[:2]
	if max_size is not None:
		target_height, target_width = fit_dimensions(target_height, target_width, max_size)

	resized_arrays = [resize_array(arr, target_width, target_height) for arr in arrays]

	# Create a shared grayscale image from the input textures using a perceptual projection.
	gray_arrays = []
	alpha_arrays = []
	for arr in resized_arrays:
		if arr.ndim == 2:
			gray = arr.astype(np.uint8)
			alpha = np.full(gray.shape, 255, dtype=np.uint8)
		elif arr.shape[2] >= 3:
			gray = np.dot(arr[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).round().astype(np.uint8)
			alpha = arr[..., 3].astype(np.uint8) if arr.shape[2] >= 4 else np.full(gray.shape, 255, dtype=np.uint8)
		else:
			gray = arr[..., 0].astype(np.uint8)
			alpha = np.full(gray.shape, 255, dtype=np.uint8)
		gray_arrays.append(gray)
		alpha_arrays.append(alpha)

	shared_gray = np.mean(np.stack(gray_arrays, axis=0), axis=0).round().astype(np.uint8)
	shared_alpha = np.mean(np.stack(alpha_arrays, axis=0), axis=0).round().astype(np.uint8)

	# Build one palette row per input texture.
	channels = resized_arrays[0].shape[2]
	palette_rows = np.zeros((len(resized_arrays), palette_limit, channels), dtype=np.uint8)
	for idx, arr in enumerate(resized_arrays):
		palette_rows[idx] = build_palette_row(arr, shared_gray, palette_limit)

	save_texture(grayscale_out_path, shared_gray, texconv, dds_format, is_grayscale=True, alpha=shared_alpha)
	save_texture(palette_out_path, palette_rows, texconv, dds_format)


def build_palette_row(color_image, shared_gray, palette_limit=256):
	"""Create a palette row that preserves overall color relationships without overfitting to a single pixel."""
	flat_color = color_image.reshape(-1, color_image.shape[2]).astype(np.float32)
	flat_gray = shared_gray.reshape(-1)
	if flat_color.shape[0] == 0:
		return np.full((palette_limit, color_image.shape[2]), 127, dtype=np.uint8)

	palette = np.zeros((palette_limit, color_image.shape[2]), dtype=np.float32)
	target_grays = np.linspace(0, 255, palette_limit).round().astype(int)
	for idx, target in enumerate(target_grays):
		mask = (flat_gray >= max(0, target - 8)) & (flat_gray <= min(255, target + 8))
		if np.any(mask):
			palette[idx] = np.mean(flat_color[mask], axis=0)
		else:
			palette[idx] = np.mean(flat_color, axis=0)

	return np.clip(np.round(palette), 0, 255).astype(np.uint8)


def load_texture_array(path, texconv):
	image = load_dds_safe(path, texconv)
	array = np.array(image)

	if array.ndim == 2:
		array = np.stack([array, array, array], axis=-1)
	elif array.ndim == 3 and array.shape[2] == 1:
		array = np.repeat(array, 3, axis=-1)
	elif array.ndim == 3 and array.shape[2] == 2:
		rgba = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
		rgba[..., :2] = array
		rgba[..., 2] = 0
		rgba[..., 3] = 255
		array = rgba
	elif array.ndim == 3 and array.shape[2] not in (3, 4):
		raise ValueError(f"Unsupported image shape {array.shape} for {path}")

	return array.astype(np.uint8)


def resize_array(array, width, height):
	image = Image.fromarray(array)
	resized = image.resize((width, height), Image.LANCZOS)
	return np.array(resized)


def fit_dimensions(height, width, max_size):
	size = max(height, width)
	while size > max_size:
		height = max(1, height // 2)
		width = max(1, width // 2)
		size = max(height, width)
	return height, width


def save_texture(out_path, image_array, texconv, dds_format, is_grayscale=False, alpha=None):
	folder = dirname(out_path)
	format_for_texconv = dds_format

	if is_grayscale:
		gray_array = image_array.astype(np.uint8)
		if gray_array.ndim == 3 and gray_array.shape[2] == 1:
			gray_array = gray_array[..., 0]
		if gray_array.ndim != 2:
			gray_array = np.mean(gray_array, axis=-1).astype(np.uint8)
		if alpha is not None:
			gray_rgba = np.zeros((gray_array.shape[0], gray_array.shape[1], 4), dtype=np.uint8)
			gray_rgba[..., 0] = gray_array
			gray_rgba[..., 1] = gray_array
			gray_rgba[..., 2] = gray_array
			gray_rgba[..., 3] = alpha.astype(np.uint8)
			png_path = splitext(out_path)[0] + '.png'
			Image.fromarray(gray_rgba, mode='RGBA').save(png_path)
		else:
			png_path = splitext(out_path)[0] + '.png'
			Image.fromarray(gray_array, mode='L').save(png_path)
		try:
			subprocess.check_call([texconv, "-nologo", "-y", "-m", "1", "-f", format_for_texconv, "-ft", "dds", "-o", folder, png_path], creationflags=0x08000000, shell=False)
		except FileNotFoundError as exc:
			raise RuntimeError(f"texconv executable not found: {texconv}") from exc
		except subprocess.CalledProcessError as exc:
			raise RuntimeError(f"texconv failed for {out_path} with exit code {exc.returncode}") from exc
		created_output = splitext(png_path)[0] + '.dds'
		if created_output != out_path:
			os.replace(created_output, out_path)
		return

	try:
		iio.imwrite(out_path, image_array, format='dds')
	except PermissionError:
		# Try again in 5 seconds. Could be a temporary issue.
		time.sleep(5)
		iio.imwrite(out_path, image_array, format='dds')

	try:
		subprocess.check_call([texconv, "-nologo", "-y", "-m", "1", "-f", format_for_texconv, "-ft", "dds", "-o", folder, out_path], creationflags=0x08000000, shell=False)
	except FileNotFoundError as exc:
		raise RuntimeError(f"texconv executable not found: {texconv}") from exc
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(f"texconv failed for {out_path} with exit code {exc.returncode}") from exc


def main():
	if len(argv) < 2:
		print("Usage:\n  RasterizeGrayScaleToPalette.py path\\to\\texconv.exe grayscale.dds palette.dds scale(0.0-1.0) out.dds 1024 BC3_UNORM\n  RasterizeGrayScaleToPalette.py --reverse path\\to\\texconv.exe grayscale_out.dds palette_out.dds 1024 BC3_UNORM input1.dds [input2.dds ...]")
		return

	if argv[1] == '--reverse':
		if len(argv) < 8:
			print("Usage: RasterizeGrayScaleToPalette.py --reverse path\\to\\texconv.exe grayscale_out.dds palette_out.dds 1024 BC3_UNORM [32|64|128|256] input1.dds [input2.dds ...]")
			return
		texconv = argv[2]
		gray_out = argv[3]
		palette_out = argv[4]
		max_size = int(argv[5])
		dds_format = argv[6]
		palette_limit = 256
		input_paths = argv[7:]
		if input_paths and input_paths[0].isdigit():
			palette_limit = int(input_paths.pop(0))
		if palette_limit not in {32, 64, 128, 256}:
			raise ValueError("palette limit must be one of: 32, 64, 128, 256")
		reverse_palette(texconv, input_paths, gray_out, palette_out, max_size, dds_format, palette_limit)
		return

	if len(argv) != 8:
		print("Usage: RasterizeGrayScaleToPalette.py path\\to\\texconv.exe grayscale.dds palette.dds scale(0.0-1.0) out.dds 1024 BC3_UNORM")
		return
	texconv = argv[1]
	gpath = argv[2]
	ppath = argv[3]
	scale = float(argv[4])
	out = argv[5]
	max_size = int(argv[6])
	dds_format = argv[7]
	apply_palette(texconv, gpath, ppath, scale, out, max_size, dds_format)


def load_dds_safe(path, texconv):
	# Try direct load (will fail for BC7)
	try:
		return Image.open(path)
	except (UnidentifiedImageError, OSError):
		pass

	# Fallback: decompress with texconv
	folder = dirname(path)
	try:
		subprocess.check_call([texconv, "-nologo", "-y", "-ft", "png", "-o", folder, path], creationflags=0x08000000, shell=False)
	except FileNotFoundError as exc:
		raise RuntimeError(f"texconv executable not found: {texconv}") from exc
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(f"texconv failed for {path} with exit code {exc.returncode}") from exc
	return Image.open(splitext(path)[0] + '.png')


if __name__ == '__main__':
	main()