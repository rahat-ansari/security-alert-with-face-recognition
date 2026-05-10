#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

try:
    from jupyter_core.paths import jupyter_data_dir
except Exception:
    jupyter_data_dir = None

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, 'static')

def main():
    if not os.path.isdir(SRC):
        print('Error: static directory not found at', SRC)
        sys.exit(1)

    try:
        if jupyter_data_dir:
            dst = os.path.join(jupyter_data_dir(), 'nbextensions', 'object_tracking')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(SRC, dst, dirs_exist_ok=True)
            print('Copied extension files to', dst)
        else:
            print('jupyter_core.paths.jupyter_data_dir not available; will attempt to use jupyter nbextension CLI')

        # Try enabling via jupyter CLI
        try:
            subprocess.check_call(['jupyter', 'nbextension', 'install', SRC, '--sys-prefix', '--overwrite'])
            subprocess.check_call(['jupyter', 'nbextension', 'enable', 'object_tracking/main', '--sys-prefix'])
            print('nbextension installed and enabled (system prefix).')
        except Exception as e:
            print('Warning: automatic enable failed:', e)
            print('You can enable manually with:')
            print('  jupyter nbextension install %s --user' % SRC)
            print('  jupyter nbextension enable object_tracking/main --user')

    except Exception as err:
        print('Installation failed:', err)
        sys.exit(2)

if __name__ == '__main__':
    main()
