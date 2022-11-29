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


