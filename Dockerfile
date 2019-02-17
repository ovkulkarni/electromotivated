FROM jjanzic/docker-python3-opencv
WORKDIR /srv
ADD ./ /srv
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install gunicorn
RUN python3 manage.py migrate
CMD /usr/local/bin/gunicorn -b 0.0.0.0:1337 -w 4 vthacks.wsgi
