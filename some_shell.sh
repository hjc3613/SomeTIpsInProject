#windows 下压缩包，在linux下去解压，解决乱码问题
LANG=C 7za x your-zip-file.zip
convmv -f GBK -t utf8 --notest -r .
# 批量修改文件名字
find . -type f -name "* *.xml" -exec bash -c 'mv "$0" "${0// /_}"' {} \;

# mongo 从硬盘恢复数据 use admin;db.auth('usename', 'passwd')
$mongorestore --host databasehost:98761 --username restoreuser
--password restorepwd --authenticationDatabase admin --db targetdb ./path/to/dump

#  清空mongo数据库
mongo --quiet --eval 'db.getMongo().getDBNames().forEach(function(i){db.getSiblingDB(i).dropDatabase()})'
