Install Webots:
https://cyberbotics.com/doc/guide/installation-procedure

Webots Command Line:  
https://cyberbotics.com/doc/guide/starting-webots#command-line-arguments

Add these environment variables

export PATH="/Applications/Webots.app/Contents/MacOS:$PATH"  
export WEBOTS_HOME=/Applications/Webots.app  
export WEBOTS_HOME_PATH=$WEBOTS_HOME/Contents  
export QT_QPA_PLATFORM_PLUGIN_PATH="/Applications/Webots.app/Contents/lib/webots/qt/plugins/platforms"  
export LD_LIBRARY_PATH="~/software/openssl-1.0.2"  
export QT_PLUGIN_PATH="/Applications/Webots.app/Contents/lib/webots/qt/plugins"  
export PYTHONPATH="/Applications/Webots.app/Contents/lib/controller/python:$PYTHONPATH"  


For Mac: You might need to install the x86_64 versions of numpy and opencv-python using these commands  
pip uninstall numpy  
pip cache purge  
arch -x86_64 pip install numpy  

Replace numpy with opencv-python as well  

For Mac: You might need to install the QT dependencies  
https://github.com/cyberbotics/webots/wiki/Qt-compilation  

For Mac: You might need to install openssl-1.0.2  
https://github.com/cyberbotics/webots/wiki/OpenSSL-compilation  
