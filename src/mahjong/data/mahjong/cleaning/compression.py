"""Lossless Zstd frames; text/3 semantics and per-game seek boundaries stay intact."""
from compression import zstd
import zlib

from .common import require
from .metrics import timed

ZSTD_STORAGE = 'zstd/1'
BLOCK_STORAGE = 'zstd-blocks/1'


def compressed_storage(catalog):
    return catalog['binding'].get('compression') in (ZSTD_STORAGE, BLOCK_STORAGE)


def product_names(catalog):
    if catalog['binding'].get('compression') == BLOCK_STORAGE:
        return 'data.blocks.zst', 'index.json.zst'
    return ('data.txt.zst', 'index.json.zst') if compressed_storage(catalog) else ('data.txt.gz', 'index.json')


def decompress_frame(data, *, zstandard, max_bytes, expected_size=None):
    """Reject extra frames, incomplete frames and expansion beyond the byte limit."""
    try:
        decoder = zstd.ZstdDecompressor() if zstandard else zlib.decompressobj(31)
        result = decoder.decompress(data, max_bytes + 1)
        require(len(result) <= max_bytes and decoder.eof and not decoder.unused_data and
                (zstandard or not decoder.unconsumed_tail) and
                (expected_size is None or len(result) == expected_size),
                'STORAGE_CORRUPT', 'compressed frame boundary/decoded size')
        return result
    except (zstd.ZstdError, zlib.error) as error:
        require(False, 'STORAGE_CORRUPT', str(error))


class ZstdWriter:
    def __init__(self, output, *, level=9):
        self.output = output
        self.compressor = zstd.ZstdCompressor(options={
            zstd.CompressionParameter.compression_level: level,
            zstd.CompressionParameter.checksum_flag: 1})

    def __enter__(self):
        return self

    def write(self, data):
        with timed('compression_seconds'):
            self.output.write(self.compressor.compress(data))

    def __exit__(self, *error):
        if error[0] is None:
            with timed('compression_seconds'):
                self.output.write(self.compressor.flush())
