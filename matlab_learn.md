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

## [~,variable]
But if you just want to know C (and you don't care IDX), it is not usefull to assign this value to a variable.
So, when you use [~,palette], that means that you just want the second output of your function, and do not care the first one.

## repmat
B = repmat(A,2)
B = 6×6

   100     0     0   100     0     0
     0   200     0     0   200     0
     0     0   300     0     0   300
   100     0     0   100     0     0
     0   200     0     0   200     0
     0     0   300     0     0   300
     
 ## usual functions
 https://zhuanlan.zhihu.com/p/343835034
 
 ## disp
 A = [15 150];
S = 'Hello World.';
Display the value of each variable.

disp(A)
    15   150
disp(S)
Hello World.

## strcmpi
s1 = {'Tinker', 'Tailor';
      '  Soldier', 'Spy'};
s2 = {'Tinker', 'Baker';
      'Soldier', 'SPY'};

tf = strcmpi(s1,s2)
tf = 2x2 logical array

   1   0
   0   1

tf(1,1) is 1 because 'Tinker' is in the first cell of both arrays. tf(2,2) is 1 because 'Spy' and 'SPY' differ only in case. tf(2,1) is 0 because ' Soldier' in s1(2,1) has whitespace characters, and 'Soldier' in s2(2,1) does not.

## sc
SC is a useful function for displaying rich image data, of use to anyone wishing to visualize and save 2D data in ways beyond that which MATLAB built-in functions allow.

## numel
numel
Number of data elements in fi array

## confidence score过程
先造一个分布（用nearest distance），然后自己想要的分布
然后，再套在一组数据里，按照分布来构建一套数据，也就是在这里，我们需要弄出一个pdf函数
 
 ## confidence score
 hcs predict file
 
