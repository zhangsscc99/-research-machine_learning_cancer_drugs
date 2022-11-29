## clear clc?
a = 1;
b = 2;
clear a
Only variable b remains in the workspace.

clc clears all the text from the Command Window, resulting in a clear screen. 
After running clc, you cannot use the scroll bar in the Command Window to see previously displayed text. 
You can, however, use the up-arrow key ↑ in the Command Window to recall statements from the command history.

## exist
exist
Check if a variable or file exists

~exist: not exist

### e.g 
if exist('AmechM','var')
if ~exist('../results','dir')

## restoredefaultpath
restoredefaultpath resets the MATLAB® search path to the factory-installed state. 
By default, the search path includes the MATLAB userpath folder, the folders defined as part of the MATLABPATH environment variable, 
and the folders provided with MATLAB and other MathWorks® products.


## addpath

addpath(folderName1,...,folderNameN,position)将指定的文件夹添加到搜索路径的顶部或底部，由 指定position。
mkdir('matlab/myfiles')   
addpath('matlab/myfiles')  
savepath matlab/myfiles/pathdef.m

## run .m file
How to run the m-file?
After the m-file is saved with the name filename.m in the current MATLAB folder or directory, you can execute the commands in the m-file by simply typing filename at the MATLAB command window prompt.

If you don't want to run the whole m-file, you can just copy the part of the m-file that you want to run and paste it at the MATLAB prompt.



## get
get(req) returns the value of all properties of the requirement object (sdo.requirements.StepResponseEnvelope, ...).

get(req,PropertyName) returns value of a specific property. Use a cell array of property names to return a cell array with multiple property values.

v = get(h)
v = get(h,propertyName)
v = get(h,propertyArray)
v = get(h,'default')
v = get(h,defaultTypeProperty)
v = get(groot,'factory')
v = get(groot,factoryTypeProperty)

## set
Modify the properties of the model. Add an input delay of 0.1 second, label the input as torque, and set the D matrix to 0.

set(sys,'InputDelay',0.1,'InputName','torque','D',0);
