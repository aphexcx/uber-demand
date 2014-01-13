uber-demand
===========

One of the most important signals in Uber's system is demand. It's used in
a variety of capacities, from calculating dynamic (surge) pricing to making
decisions about the number of drivers needed to keep the system running
smoothly. One of the ways Uber quantifies demand is by tracking when users
open the Uber app on their phones.

This service predicts future demand based on historical data.

Setup:
1) Install MayaVi:
    a) Grab VTK and wxPython:
        apt-get python-vtk python-wxgtk2.8 (debian/ubuntu)
        yum install vtk-python wxPython-devel (centos/rhel)
    b) On Centos, mayavi works better if you install it from yum:
        yum install Mayavi
       Otherwise, just use pip:
        pip install wxpython mayavi

2) Install requirements.txt
        pip install -r requirements.txt

3) Start redis:
        redis-server
    and run the celery worker with:
        celery worker --app=tasks --loglevel=info

You're ready to run!
        python main.py
