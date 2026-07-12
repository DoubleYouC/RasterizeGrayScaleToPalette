import subprocess
from sys import argv
import numpy as np
import imageio.v3 as iio
from PIL import Image, UnidentifiedImageError
from os.path import splitext, dirname

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

	iio.imwrite(out_path, final_image, format='dds')

	folder = dirname(out_path)

	subprocess.check_call([texconv, "-nologo", "-y", "-m", "1", "-f", dds_format, "ft", "dds", "-o", folder, out_path], shell=False)

def main():
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
	except UnidentifiedImageError:
		pass

	# Fallback: decompress with texconv
	folder = dirname(path)
	subprocess.check_call([texconv, "-nologo", "-y", "-ft", "png", "-o", folder, path], shell=False)
	return Image.open(splitext(path)[0] + '.png')



if __name__ == '__main__':
	main()