//chess code


//https://www.onlinegdb.com/online_c++_compiler
/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>     // std::cout
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* abs */
#include <algorithm>    // std::min

using namespace std;


int signfunc(int val)
{
	int signval;
	
	if(val < 0)
	{
		signval = -1;
	}
	else
	{
		signval = 1;
	}
	return signval;
}

// -----------------------------------

void intpos(int x, int y, int n, int &front, int &back, int &left, int &right)
{
	front = x;  //n-y-1;
    back = (n-1)-x; //y;
    left = y; //x;
    right = (n-1)-y;  //n-x-1;
	printf("front : %d\n", front);
	printf("back : %d\n", back);
	printf("left : %d\n", left);
	printf("right : %d\n", right);
}

// -----------------------------------

int visual(int xarr, int yarr, int n, int nsize)
{
    int i, j, a[n][n];
    
    printf("xarr : %d\n", xarr);
	printf("yarr : %d\n", yarr);
    
    // plot the board
    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            
            if(i == xarr && j == yarr)
            {
                a[xarr][yarr] = 4;
                cout << a[xarr][yarr];
            }
            else
            {
                a[i][j] = 0;
                cout << a[i][j];
            }
        }
        cout << " " << endl;
    }
    return 0;
}

// -----------------------------------

void straight(int b_x, int b_y, int x, int y, int n, int &xnew, int &ynew)
{
    int front, back, left, right;
	intpos(x, y, n, front, back, left, right);
	
	
	if(signfunc(b_y) == -1)
	{
		if(abs(b_y) > left)  // gauche
		{
			b_y = -left;
		}
	}
	else if(signfunc(b_y) == 1)
	{
		if(abs(b_y) > right)
		{
			b_y = right;
		}
	}
	
	if(signfunc(b_x) == -1)
	{
		if(abs(b_x) > back)
		{
			b_x = -back;
		}
	}
	else if(signfunc(b_x) == 1)
	{
		if(abs(b_x) > front)
		{
			b_x = front;
		}
	}
	
	printf("b_x : %d\n", b_x);
	printf("x : %d\n", x);
	printf("b_y : %d\n", b_y);
	printf("y : %d\n", y);
	
	xnew = x - b_x;
    ynew = y + b_y;
    
    printf("xnew : %d\n", xnew);
	printf("ynew : %d\n", ynew);
}

// -----------------------------------

void diagonal_hd(int b_hd, int x, int y, int n, int &xnew, int &ynew)
{
    int front, back, left, right;
	intpos(x, y, n, front, back, left, right);
	
	if(signfunc(b_hd) == -1)
	{
		if(abs(b_hd) > left | abs(b_hd) > back)
		{
            b_hd = -min(left, back); // new limited command
		}
		xnew = x - b_hd; //want to increase x, so cancel negative
		ynew = y + b_hd;  //want to decrease y
	}
	else if(signfunc(b_hd) == 1) // droit
	{
		if(b_hd > right | b_hd > front)
		{
            b_hd = min(right, front); // new limited command
		}
		xnew = x - b_hd; //want to decrease x
        ynew = y + b_hd; //want to increase y
	}
	
}

// -----------------------------------

void diagonal_bd(int b_bd, int x, int y, int n, int &xnew, int &ynew) // down_right_diag
{
    int front, back, left, right;
	intpos(x, y, n, front, back, left, right);
    
    if(signfunc(b_bd) == -1)   // we want to move bas_droit
	{
        if(abs(b_bd) > left | abs(b_bd) > front)
		{
            b_bd = -min(left, front); // new limited command
		}
		xnew = x + b_bd; //want to decrease x
		ynew = y + b_bd;  //want to decrease y
	}
    else if(signfunc(b_bd) == 1) // we want to move the opposite of physical bas_droit
	{
        if(b_bd > right | b_bd > back)
		{
            b_bd = min(right, back); // new limited command
		}
		xnew = x + b_bd; //want to increase x
		ynew = y + b_bd;  //want to increase y
	}
    
	//printf("xnew : %d\n", xnew)
    //printf("ynew : %d\n", ynew)
}

// -----------------------------------

int main()
{
    int n = 8, veclen=0;
    // int *tmp1; // to free ex allocated arrays
    // int *tmp2;
    string command;
    int dist, dia;
    int b_x, b_y, b_hd, b_bd;
    
    int nsize = 1;
    
    //start position
    int x = n-1;
    int y = 4;
    
    //int *xarr = (int*)malloc(nsize * sizeof(int));
    //int *yarr = (int*)malloc(nsize * sizeof(int));
	// int *xarr = new int[nsize];
    // int *yarr = new int[nsize];
    int xarr=x, yarr=y;
    int xout=0, yout=0;
    
    xarr = x;
    yarr = y;
    
    veclen = veclen + 1;
    visual(xarr, yarr, n, nsize);
    
	// -----------------------------------
    
	
    int max_commands = 10;
    for(int q=0;q<max_commands;q++)
    {
        cout << "What is the next move?: b_x, b_y, b_hd, b_bd, circle : ";
        cin >> command;
        cout << "Entrez le nombre d'espaces à déplacer : ";
        cin >> dist;
        
        if(command == "b_x")
        {
            // -----------------------------------
			
			
			//pass new value to last entry of xarr
			b_x = dist;
			b_y = 0;
			int xout=0, yout=0;
			straight(b_x, b_y, xarr, yarr, n, xout, yout);
			printf("xout : %d, yout : %d", xout, yout);
			xarr = xout;
			yarr = yout;
			printf("veclen : %d", veclen);
			visual(xarr, yarr, n, veclen);
			// -----------------------------------
        }
        else
        {
            if(command == "b_y")
            {
                // -----------------------------------
				
				//pass new value to last entry of xarr
				b_x = 0;
			    b_y = dist;
				straight(b_x, b_y, xarr, yarr, n, xout, yout);
			    printf("xout : %d, yout : %d", xout, yout);
    			xarr = xout;
    			yarr = yout;
    			printf("veclen : %d", veclen);
    			visual(xarr, yarr, n, veclen);
				// -----------------------------------
            }
            else
            {
                if(command == "b_hd")
                {
					// -----------------------------------
					//pass new value to last entry of xarr
					b_hd = dist;
					diagonal_hd(b_hd, xarr, yarr, n, xout, yout);
					printf("xout : %d, yout : %d", xout, yout);
        			xarr = xout;
        			yarr = yout;
        			printf("veclen : %d", veclen);
        			visual(xarr, yarr, n, veclen);
					// -----------------------------------
                }
                else
                {
                    if(command == "b_bd")
                    {
						// -----------------------------------
						
						//pass new value to last entry of xarr
						b_bd = dist;
						diagonal_bd(b_bd, xarr, yarr, n, xout, yout);
						printf("xout : %d, yout : %d", xout, yout);
            			xarr = xout;
            			yarr = yout;
            			printf("veclen : %d", veclen);
            			visual(xarr, yarr, n, veclen);
						// -----------------------------------
                    }
                    else
                    {
                        if(command == "circle")
                        {
                            dia = dist;
                            b_x = 0;
                            b_y = dia;
                        	straight(b_x, b_y, xarr, yarr, n, xout, yout);
                        	xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        
                            b_hd = dia;
                            diagonal_hd(b_hd, xarr, yarr, n, xout, yout);
                            xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        	
                            b_x = dia;
                            b_y = 0;
                            straight(b_x, b_y, xarr, yarr, n, xout, yout);
                            xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        	
                            b_bd = -dia;
                            diagonal_bd(b_bd, xarr, yarr, n, xout, yout);
                            xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        	
                            b_x = 0;
                            b_y = -dia;
                        	straight(b_x, b_y, xarr, yarr, n, xout, yout);
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        	
                            b_hd = -dia;
                            diagonal_hd(b_hd, xarr, yarr, n, xout, yout);
                            xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        
                            b_x = -dia;
                            b_y = 0;
                            straight(b_x, b_y, xarr, yarr, n, xout, yout);
                            xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        	// -----------------------------------
                        
                            b_bd = dia;
                            diagonal_bd(b_bd, xarr, yarr, n, xout, yout);
                            xarr = xout;
            			    yarr = yout;
                        	visual(xarr, yarr, n, veclen);
                        }
                    }
                }
            }
        }
    }
    
    return 0;
}