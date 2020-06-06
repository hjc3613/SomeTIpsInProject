#windows 下压缩包，在linux下去解压，解决乱码问题
LANG=C 7za x your-zip-file.zip
convmv -f GBK -t utf8 --notest -r .
# 批量修改文件名字
find . -type f -name "* *.xml" -exec bash -c 'mv "$0" "${0// /_}"' {} \;
