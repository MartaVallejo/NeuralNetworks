clc;
a = -5.09346;
b = abs(a)

c = 0.09346;
d = abs(c)

a = zeros(5,2);

a = randi([2, 1]);
disp(a);

for i = 5:-1:1
   disp(i); 
end

x = 2*rand(1)-1;
smallest = 1;
biggest = -1;
for i = 1:1000000
    x = 2*rand(1)-1;
    if(x < smallest)
        smallest = x;
    end
    if(x > biggest)
        biggest = x;
    end
end

fprintf('Smallest: %d Biggest: %d', smallest, biggest);

fx1 = 1./(1 + exp(-x)); %Binary
fx2 = -1 + 2./(1 + exp(-x)); %Bipolar
fx3 = logsig(x);
fx4 = tanh(x);
fx5 = tansig(x);
fx6 = 1 - (tanh(x) * tanh(x));
fx7 = 1.0 - x * x;
fx8 = diff(fx4);
syms f(x1)
f(x1) = diff(tanh(x1));
x1 = x;
a = f(x1);

element1 = Element('OK');

disp(element1.value);

%reshape(A,5,0);
%r = zeros(1,3);
%r = char.empty(1,7);
r = nan(0,7);
r(1,:) = 'val1';
r(2) = 'val2';
r(3) = 'val3';

r = getResult();
testvar = r(1);
testvar2 = r(2);
testvar3 = r(length(r));
testvar4 = r(2);
alphabet = (['0':'z']);
num = 0;
myBool = 0;
while ~myBool
   
    fprintf('OK');
    num = num + 1;
    if(num > 10)
       myBool = 1; 
    end
end

function result = getResult()
result = {};
    result = [result, 'hola'];
result = [result, 'pos2'];
result = [result, 'pos3'];
end

function testfunc(obj, arg2)
           var1 = testfunc2(obj, arg2, num2str(10));
           disp(var1);
        end
        
        function result = testfunc2(obj, arg1, arg2)
           result = strcat(arg1, arg2);
        end

