#windows 下压缩包，在linux下去解压，解决乱码问题
LANG=C 7za x your-zip-file.zip
convmv -f GBK -t utf8 --notest -r .
