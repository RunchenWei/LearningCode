#include <stdio.h>
#include <iostream>

struct Node
{
    int status;
    int threshold;
};


int main()
{
    int n,p;
    scanf("%d %d",&n,&p);
    Node n1[n];
    int count=0;
    for(int i = 0; i < n; i++)
    {
        scanf("%d %d",&n1[i].status,&n1[i].threshold);
        if (n1[i].status!=0) {
            count++;
        }
    }
    int height = count+1;
    int width= p/height;
    int edge[height][width];
    for(int i = 0; i < height; i++)
    {
        for(int  j = 0; j < width; j++)
        {
            edge[i][j]=0;
        }
    }
    
    int a,b,c;
    for(int i = 0; i < p; i++)
    {
        scanf("%d %d %d",&a,&b,&c);
        edge[a-1][b-1]=c;
    }
    int flag=0;
    for(int i = 0; i < n; i++)
    {
        if (n1[i].status!=0) 
        {
            continue;
        }
        else
        {
            int sum=0;
            for(int j = 0; j < n; j++)
            {
                if (n1[j].status!=0) {
                    sum=sum+n1[j].status*edge[j][i];
                }
            }
            sum=sum-n1[i].threshold;
            if(sum>0)
            {
                printf("%d %d\n",i+1,sum);
                flag=1;
            }
        }
        
    }
    if(flag==0)
    {
        printf("NULL");
    }
    
    
    // 前面非0的是输入
    // 前面是1的是隐层
    system("pause");
    return 0;
}