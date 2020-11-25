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

# rsync命令
这会将文件夹A放入文件夹B：
rsync -avu --delete "/home/user/A" "/home/user/B" 
如果希望文件夹A和B的内容相同，则将其/home/user/A/（带有斜线）作为源。这不占用文件夹A，而是所有内容，并将其放入文件夹B。像这样：
rsync -avu --delete "/home/user/A/" "/home/user/B"
-a 进行同步以保留所有文件系统属性
-v 冗长地跑
-u 仅复制修改时间较新的文件（如果时间相等，则复制大小不同的文件）
--delete 删除目标文件夹中源文件中不存在的文件

# mysql 8.0允许所有ip访问
mysql> CREATE USER 'root'@'%' IDENTIFIED BY 'root';
mysql> GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
