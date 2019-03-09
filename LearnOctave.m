for i = 1:10,  % for loops
  v(i) = 2^i;
endfor;
v

indices = 1:10;

for i = indices,  
  disp(i);
endfor;
v
v = v'

% while statements
i = 1;  
while i <= 5, 
  v(i) = 100;
  i = i + 1;
endwhile;

v

% while statements
i = 1; 
while true, 
  v(i) = 999; 
  i = i +1; 
  if i == 6, 
    break;
  endif
endwhile
v

% if statements
v(1) = 2; 

if v(1) == 1, 
  disp("the value is one");
elseif v(1) == 2, 
  disp("the value is two");
else
  disp("the value is not one or two"); 
endif


% write some cost function 

X = [1 1; 1 2; 1 3];

y = [1; 2; 3];

theta = [0; 0];

j = costFunctionJ(X, y, theta)


