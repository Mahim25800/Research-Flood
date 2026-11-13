"""Read selected members of the public MMFlood ZIP via bounded HTTP ranges."""
import io
import re
import struct
import zlib
from zipfile import ZipFile

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.transferred = 0
        self.size = self._fetch(0, 0)[1]

    def _fetch(self, start, end):
        expected = end - start + 1
        for attempt in range(3):
            try:
                # Distinct cache keys prevent a proxy from returning a previous range.
                with requests.get(self.url, params={'px1_range': f'{start}-{end}'},
                                  headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(10, 25)) as response:
                    match = re.fullmatch(rf'bytes {start}-{end}/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status_code != 206 or not match:
                        raise OSError(f'Unexpected ZIP range response: HTTP {response.status_code}, '
                                      f'{response.headers.get("Content-Range")}')
                    data = response.raw.read(expected + 1)
                    if len(data) != expected:
                        raise OSError(f'Truncated or oversized ZIP range: {len(data)} != {expected}')
                    self.transferred += len(data)
                    return data, int(match[1])
            except (requests.RequestException, OSError) as exc:
                if attempt == 2:
                    raise OSError(f'MMFlood ZIP range {start}-{end} failed') from exc

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def read(self, size=-1):
        available = max(0, self.size - self.position)
        size = available if size < 0 else min(size, available)
        if size == 0:
            return b''
        if size > 128 * 1024**2 or self.transferred + size > 512 * 1024**2:
            raise OSError('MMFlood smoke download limit exceeded')
        data, total = self._fetch(self.position, self.position + size - 1)
        if total != self.size:
            raise OSError('MMFlood ZIP size changed during read')
        self.position += size
        return data


def read_member(source, archive, name):
    entry = archive.getinfo(name)
    if entry.compress_type != 9:
        return archive.read(entry)
    import inflate64

    source.seek(entry.header_offset)
    header = source.read(30)
    if header[:4] != b'PK\x03\x04' or entry.flag_bits & 1:
        raise ValueError(f'Unsupported ZIP member: {name}')
    filename_size, extra_size = struct.unpack_from('<HH', header, 26)
    source.seek(filename_size + extra_size, io.SEEK_CUR)
    data = inflate64.Inflater().inflate(source.read(entry.compress_size))
    if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
        raise ValueError(f'ZIP member integrity check failed: {name}')
    return data


if __name__ == '__main__':
    import json
    from mm_prep import URL
    with RangeFile(URL) as source, ZipFile(source) as archive:
        names = archive.namelist()
        print(len(names), 'members; first entries:', names[:12], flush=True)
        print('activation metadata:', [n for n in names if n.endswith('activations.json')], flush=True)
        print('first SAR files:', [n for n in names if '/s1_raw/' in n and n.endswith('.tif')][:5], flush=True)
        print('ZIP compression methods:', {entry.compress_type for entry in archive.infolist()}, flush=True)
        meta = json.loads(read_member(source, archive, 'activations.json'))
        print(type(meta).__name__, str(meta)[:1200], flush=True)
