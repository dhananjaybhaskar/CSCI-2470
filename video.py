import sys
import subprocess
import os

path = sys.argv[1]
path = path + "/"

output = sys.argv[2]

from subprocess import call
call(["ffmpeg", "-framerate", "60", "-i",
    path + "Image%04d.png", "-vb", "20M",
    output])