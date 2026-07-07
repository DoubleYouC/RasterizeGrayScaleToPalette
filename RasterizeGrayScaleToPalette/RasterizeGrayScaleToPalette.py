from sys import argv
import numpy as np
import imageio.v3 as iio
import os
from tempfile import mkdtemp
from subprocess import check_call, CalledProcessError

def apply_palette(texconv, grayscale_path, palette_path, scale, out_path):
	"""Apply a palette row to a grayscale image.

	grayscale_path: path to grayscale
	palette_path: path to palette (each row is a palette)
	scale: 0.0..1.0 selects row in palette
	out_path: output path
	"""
	# read images
	g = iio.imread(grayscale_path)
	p = iio.imread(palette_path)

	# ensure grayscale is single channel 0..255
	if g.ndim == 3:
		# assume RGB where channels are equal
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
	row = int(np.clip(scale, 0.0, 1.0) * (ph - 1))
	palette_row = palette_rgb[row]  # shape (width, 3)

	# if palette width != 256 and grayscale uses 0..255, scale indices
	# map grayscale values (0..255) to columns 0..pw-1
	indices = (g.astype(np.float32) / 255.0 * (pw - 1)).round().astype(int)

	# build colored image
	h, w = g.shape
	out = np.zeros((h, w, 3), dtype=np.uint8)
	out[..., 0] = palette_row[indices, 0]
	out[..., 1] = palette_row[indices, 1]
	out[..., 2] = palette_row[indices, 2]

	# write temp file
	tmpdir = mkdtemp()
	tmp_png = os.path.join(tmpdir, 'tmp_out.png')

	iio.imwrite(tmp_png, out, format='png')

	out_dir = os.path.dirname(os.path.abspath(out_path)) or os.getcwd()
	try:
		check_call([texconv, '-ft', 'dds', '-f', 'BC3_UNORM', '-y', '-o', out_dir, tmp_png])
	except CalledProcessError as e:
		print('texconv failed:', e)
		return 1

	# texconv writes tmp_out.dds in out_dir; move/rename to requested out_path
	base_name = os.path.splitext(os.path.basename(tmp_png))[0] + '.dds'
	produced = os.path.join(out_dir, base_name)
	if os.path.exists(produced):
		os.replace(produced, out_path)
		print('Wrote', out_path)
		return 0
	else:
		print('Expected output not found. Check texconv output in', out_dir)
		return 1


def main():
	if len(argv) != 6:
		print("Usage: script.py path\to\texconv.exe grayscale.bmp palette.bmp scale(0.0-1.0) out.png")
		return
	texconv = argv[1]
	gpath = argv[2]
	ppath = argv[3]
	scale = float(argv[4])
	out = argv[5]
	apply_palette(texconv, gpath, ppath, scale, out)


if __name__ == '__main__':
	main()