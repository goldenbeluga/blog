![Image Alt 텍스트]({{site.url}}/assets/img/1.jpg )
![Image Alt 텍스트](http://blog.jaeyoon.io/assets/img/1.jpg)
![Image Alt 텍스트]({{"/assets/img/1.jpg"| relative_url}})
![Image Alt 텍스트](/assets/img/1.jpg)

# remote될 컴터 설정  
## 1. 우분투 ssh설정 
우분투 터미널을 키고 아래 명령어를 복붙한다.   
```
$ dpkg -l | grep openssh$ sudo apt-get update  
$ sudo apt-get install openssh-server  
$ dpkg -l | grep openssh #정상 설치 확인(openssh-server, openssh-sftp-server)  
$ sudo service ssh start$ service --status-all | grep + #여기에 ssh가 있으면 된 것  
$ sudo netstat -antp # SSH 서비스가 몇 번 포트를 점유하고 있는지도 확인
```
## 2. jupyter setting
![]({{site.url}}/assets/img/1.jpg )
```
$ jupyter notebook --generate-config
```
이 명령어 치면 파일이 하나 생성됨  
이 폴더로 가서 설정을 바꿔야함  
![]({{site.url}}/assets/img/2.jpg )
```
$ jupyter notebook password
```
비밀번호 설정도 하자 1234여기 들어가면 암호화된 비번이 있는데 이 비번을 나중에 입력해도 되고 귀찮으면 입력한 애를 넣어도 됨.  

![]({{site.url}}/assets/img/3.jpg )
![]({{site.url}}/assets/img/4.jpg )
 여기서 sshd의 포트가 1111 확인가능  
 나중에 원격 접속할 때 사용함 ip확인하자  
![]({{site.url}}/assets/img/5.jpg )
```
$ vim /home/user/.jupyter/jupyter_notebook_config.py
```
로 들어가서 아래 부분에 내용 추가
```
c.NotebookApp.password = u'your password'c.NotebookApp.port = 8888c.NotebookApp.ip = 'your IP(sever)'c.NotebookApp.open_browser = Falsec.NotebookApp.allow_remote_access = Truec.NotebookApp.password_required = Truec.NotebookApp.allow_origin = '' # 주석풀기
```
# remote할 컴터 설정
## putty
![]({{site.url}}/assets/img/6.jpg )
앞에서 확인한 IP와 PORT를 넣고 저장한 후 Load를 눌러주면  
터미널로 접속된다. 
remote될 컴퓨터 로그인할 때 유저아이디와 비번을 치면 접속 끝!  
## MobaxTermusername 
![]({{site.url}}/assets/img/7.jpg )
user name 적어주면 로그인 유저아이디 안물어 본다.  

# 혹시 안 될 시에는 방화벽을 내려보자.. 
$ systemctl stop firewalld 
