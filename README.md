# ArUcoMarkers

Steps to execute aruco generation and detection codes:

Make sure python is installed on your laptop.

DON'T install opencv.Only install opencv-contrib-python.


Go to Visual Studio Code,click Ctrl+Shift+P,search Python:Create Environment and create a new virtual environment.Click on .venv and wait for it to create.

Run the following commands:

pip install --upgrade pip


pip install numpy


pip install opencv-contrib-python==4.6.0.66


Then,click on Ctrl+Shift+ ~ and Clone this repository to your system by 

git clone https://github.com/HemanthK-12/ArUcoMarkers.git

Open this repository in Visual Studio Code,and click on Ctrl+Shift+~ to open terminal.Make sure you are getting the final directory as your folder.If not , navigate to that folder by running 
cd "filepath"

Click the drop down button in the newly opened terminal after clicking Ctrl+Shift+~ in the top right region and select command prompt.

Run the following commands 

python generateAruco.py


python arucoDetection.py




Happy Coding !:)





Create 
