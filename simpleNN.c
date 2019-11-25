#include <stdio.h>
#include <windows.h>


struct Node
{
    int status;
    int threshold;
    int leaf;
    int root;
};


int main()
{
    int n,p;
    scanf("%d %d",&n,&p);
    struct Node n1[n];
    for(int i = 0; i < n; i++)
    {
        scanf("%d %d",&n1[i].status,&n1[i].threshold);
        n1[i].leaf=0;
        if(n1[i].status==0)
        {
            n1[i].root=0;
        }
        else
        {
            n1[i].root=1;
        }
        
    }
    
    int edge[n][n];
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < n; k++)
        {
            edge[j][k]=0;
        }
    }

    int a,b,c=0;
    for (int m = 0; m < p; m++)
    {
        scanf("%d %d %d",&a,&b,&c);
        edge[a-1][b-1]=c;
        n1[a-1].leaf=1;
    }
    int sum;
    for (int s = 0; s < n; s++)
    {
        sum=0;
        if (n1[s].status==0)
        {
            for (int t = 0; t < n; t++)
            {
                if (n1[t].status>0)
                {
                    sum=sum+edge[t][s]*n1[t].status;
                    //printf("t:%d sum:%d   ",t,sum);
                }
            }
            sum = sum - n1[s].threshold;
            //printf("s:%d sum:%d\n",s,sum);
            n1[s].status=sum;
        }
    }
    int flag=0;
    for (int i = 0; i < n; i++)
    {
        if ((n1[i].status>0)&&(n1[i].leaf==0)&&(n1[i].root!=1))
        {
            printf("%d %d\n",i+1,n1[i].status);
            flag=1;
        }
        if ((n1[i].leaf==0)&&(n1[i].root==1))
        {
            printf("%d %d\n",i+1,n1[i].status);
            flag=1;
        }
        
        
    }
    if(flag==0)
    {
        printf("NULL\n");
    }
    
    system("pause");
    return 0;
}
