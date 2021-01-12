# -*- coding: utf-8 -*-
import os
import json
import ffmpeg_streaming
from ffmpeg_streaming import Formats, Bitrate, Representation, Size


def clean_and_mkdir(dirname):
    os.makedirs(dirname, exist_ok=True)
    files = [f for f in os.listdir(dirname)]
    for f in files:
        os.remove(os.path.join(dirname, f))


def hls(*res):
    hls_dir = 'hls'
    clean_and_mkdir(hls_dir)
    codec_options = {
        'g': 10
    }
    hls = video.hls(Formats.h264(video='libx264', audio='aac', **codec_options), hls_time=1)
    hls.representations(*res)
    hls.output(os.path.join(hls_dir, 'hls.m3u8'))


if __name__ == '__main__':
    config = json.load(open('config.json', 'r'))
    video = ffmpeg_streaming.input(f'http://localhost:{config["flask_port"]}/video_feed')

    _144p  = Representation(Size(256, 144), Bitrate(95*1024, 64*1024))
    _240p  = Representation(Size(426, 240), Bitrate(150*1024, 94*1024))
    _360p  = Representation(Size(640, 360), Bitrate(276*1024, 128*1024))
    _480p  = Representation(Size(854, 480), Bitrate(750*1024, 192*1024))
    _720p  = Representation(Size(1280, 720), Bitrate(2048*1024, 320*1024))
    _1080p = Representation(Size(1920, 1080), Bitrate(4096*1024, 320*1024))
    _2k    = Representation(Size(2560, 1440), Bitrate(6144*1024, 320*1024))
    _4k    = Representation(Size(3840, 2160), Bitrate(17408*1024, 320*1024))

    hls(_360p, _720p)
