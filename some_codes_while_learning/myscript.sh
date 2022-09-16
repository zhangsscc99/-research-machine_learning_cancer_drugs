#! /bin/bash

#ECHO COMMAND
echo Hello world!!!!!

#Variables

NAME="Brad"
echo "my name is $NAME"
#echo "my name is ${NAME}"

#user input
read -p "Enter your name:" NAME
echo "Hello $NAME, nice to meet you "

if [ "$NAME" == "Brad" ]
then 
  echo "Your name is Brad"
elif [ "$NAME" == "Jack" ]
then
  echo "your name is Jack"

else
  echo "Your name is NOT Brad"
fi

NUM1=31
NUM2=5
if [ "$NUM1" -gt "$NUM2" ]
then
    echo "$NUM1 is greater than $NUM2"
else
    echo "$NUM1 is less than $NUM2"
fi

#file conditions
FILE="test.txt"
if [ -f "$FILE" ]
then
    echo "$FILE is a file"
else
    echo "$FILE is NOT a file"
fi
