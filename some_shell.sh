#windows 下压缩包，在linux下去解压，解决乱码问题
LANG=C 7za x your-zip-file.zip
convmv -f GBK -t utf8 --notest -r .
# 批量修改文件名字
find . -type f -name "* *.xml" -exec bash -c 'mv "$0" "${0// /_}"' {} \;

#################################################mongo问题############################################################
# mongo 从硬盘恢复数据 use admin;db.auth('usename', 'passwd')
$mongorestore --host databasehost:98761 --username restoreuser
--password restorepwd --authenticationDatabase admin --db targetdb ./path/to/dump

#  清空mongo数据库
mongo --quiet --eval 'db.getMongo().getDBNames().forEach(function(i){db.getSiblingDB(i).dropDatabase()})'
pymongo admin 认证
import pymongo 
client = pymongo.MongoClient('127.0.0.1', 27017)
 
#连接admin数据库,账号密码认证
db = client.admin
db.authenticate("用户名", "密码"，"认证机制【可省略】")
 
#认证结束
db = client.mydb   # mydb数据库
col = db['mycol']  # mycol集合

######################################################## 常用bash ########################################################

# rsync命令
这会将文件夹A放入文件夹B：
rsync -avu --delete "/home/user/A" "/home/user/B" 
如果希望文件夹A和B的内容相同，则将其/home/user/A/（带有斜线）作为源。这不占用文件夹A，而是所有内容，并将其放入文件夹B。像这样：
rsync -avu --delete "/home/user/A/" "/home/user/B"
-a 进行同步以保留所有文件系统属性
-v 冗长地跑
-u 仅复制修改时间较新的文件（如果时间相等，则复制大小不同的文件）
--delete 删除目标文件夹中源文件中不存在的文件

# uniq 一定要配合sort只用！！！！！！！！！！！！！！！！！
uniq -u 只打印单一行
uniq -d 只打印重复行
uniq -c 打印去重后的行，并附带重复数量，单一行页包括在内，配合sort -rn，按重复数量降序排列
uniq 单独使用，仅仅是去重
########################################################## mysql #########################################################################
# mysql 8.0允许所有ip访问
mysql> CREATE USER 'root'@'%' IDENTIFIED BY 'root';
mysql> GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;

######################################################## hive-spark-hadoop ####################################################################### 
hive-hdfs-spark踩坑记：提交pyspark程序时，遇到bug:Unable to instantiate org.apache.hadoop.hive.ql.metadata.SessionHiveMetaStoreClient,因为spark通过hive-site.xml去读取mysql,
因此要将hive/lib/下的mysql-connector*.jar 拷贝到spark/jars/下边。同时要开启hive meta service:hive --service metastore。hive中的hive-site.xml也要放到spark/conf/里面，否则会报错：database *** not found
spark-submit --master yarn --deploy-mode cluster --conf spark.yarn.dist.archives=hdfs://localhost:9000/python_env/python36.zip#py3env --conf spark.pyspark.python=./py3env/python36/bin/python3  /home/hujunchao/Apps/spark-2.4.7-bin-hadoop2.7/examples/src/main/python/pi.py 10

pyspark-shell踩坑：将hive/lib/hive-hcatalog-core-2.3.7.jar 拷贝到 spark/jars/下，开启hive meta service： --service metastore

################# python re.sub 使用 ####################################
string = 'A23G4HFD567'
print(re.sub('(\d+)', '\g<1>  ', string, count=2))#A23  G4  HFD567

def double(matched):
    print('matched: ', matched)
    print("matched.group('value'): ", matched.group('value'))
    value = int(matched.group('value'))
    return str(value * 2)


string = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, string))
##################################################happybase thrift 连接hbase 坑################################################
   在hbase-site.xml中添加如下配置，因为默认的有链接时长限制
   <property>
    <name>hbase.thrift.server.socket.read.timeout</name>
    <value>864000000</value>
    <description>eg:milisecond</description>
  </property>

  <property>
    <name>hbase.thrift.connection.max-idletime</name>
    <value>864000000</value>
  </property>


##########################################################################
