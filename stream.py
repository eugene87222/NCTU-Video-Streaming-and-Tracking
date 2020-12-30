# -*- coding: utf-8 -*-
import ffmpeg_streaming
from ffmpeg_streaming import Formats, Bitrate, Representation, Size


def dash():
    pass

def hls():
    pass


if __name__ == '__main__':
    video = ffmpeg_streaming.input('/dev/video0', capture=True)

    _144p  = Representation(Size(256, 144), Bitrate(95 * 1024, 64 * 1024))
    _240p  = Representation(Size(426, 240), Bitrate(150 * 1024, 94 * 1024))
    _360p  = Representation(Size(640, 360), Bitrate(276 * 1024, 128 * 1024))
    _480p  = Representation(Size(854, 480), Bitrate(750 * 1024, 192 * 1024))
    _720p  = Representation(Size(1280, 720), Bitrate(2048 * 1024, 320 * 1024))
    _1080p = Representation(Size(1920, 1080), Bitrate(4096 * 1024, 320 * 1024))
    _2k    = Representation(Size(2560, 1440), Bitrate(6144 * 1024, 320 * 1024))
    _4k    = Representation(Size(3840, 2160), Bitrate(17408 * 1024, 320 * 1024))

    # hls = video.hls(Formats.h264())
    # hls.representations(_360p, _480p, _720p)
    # # hls.fragmented_mp4()
    hls = video.hls(Formats.h264(), hls_list_size=10, hls_time=5)
    hls.flags('delete_segments')
    hls.representations(_480p)
    hls.output('./hls/hls.m3u8')

    # dash = video.dash(Formats.h264())
    # # dash.representations(_144p, _240p, _360p, _480p, _720p, _1080p, _2k, _4k)
    # # dash.representations(_360p, _480p, _720p, _1080p)
    # dash.representations(_144p)
    # dash.output('./dash/dash.mpd')
