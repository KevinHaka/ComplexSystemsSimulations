#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

struct Vec_and_Mat {
    int **field, *L;
    long long int *S;
};

#define random_num ((float)rand() / (float)RAND_MAX)
void field_creator(float p, int N, struct Vec_and_Mat *vm);
int CMLT(int N, struct Vec_and_Mat *vm);

int main() {
    srand(time(NULL));

    int k, N, NMC, im_max, seeds[omp_get_num_threads()];
    long long int mmax;
    float p, temp, Pmax=0, Iav=0, I=0;

    scanf("%d", &NMC);
    scanf("%d", &N);
    scanf("%f", &p);

    for (size_t i=0; i<omp_get_num_threads(); i++) seeds[i] = rand();

    #pragma omp parallel shared(NMC,N,p, seeds) \
                        private(k,im_max,mmax,temp) \
                        reduction(+:I,Iav,Pmax) \
                        default(none) 
    {
        struct Vec_and_Mat vm;
        srand(seeds[omp_get_thread_num()]);

        vm.S = NULL;
        vm.L = NULL;
        vm.field = (int**)malloc(N*sizeof(int*));
        for (size_t n=0; n<N; n++)
            vm.field[n] = (int*)malloc(N*sizeof(int*));

        #pragma omp for
        for (size_t n=0; n<NMC; n++) {
            field_creator(p, N, &vm);
            k = CMLT(N, &vm);

            temp = 0;
            mmax = 0;
            im_max = 0;

            for (size_t idx=0; idx<k; idx++) {
                temp += vm.S[idx]*vm.S[idx]/(p*N*N);

                if (vm.S[idx] == mmax)
                    im_max++;

                else if (vm.S[idx] > mmax) 
                    mmax = vm.S[idx];
                    im_max = 1;
            }

            Pmax += mmax/(p*N*N);
            I += temp;
            Iav += (temp - im_max*mmax*mmax/(p*N*N));
        }

        for (size_t n=0; n<N; n++)
            free(vm.field[n]);
        free(vm.field);
        free(vm.L);
        free(vm.S);
    }

    Pmax /= NMC; 
    I /= NMC;
    Iav /= NMC;
    printf("%f\n%f\n%f", I, Iav, Pmax);

    return 0;
}


void field_creator(float p, int N, struct Vec_and_Mat *vm) {
    for (size_t x = 0; x < N; x++) {
        for (size_t y = 0; y < N; y++) {
            if (random_num < p) vm->field[x][y] = 1;
            else vm->field[x][y] = 0;
        }
    }
}


int CMLT(int N, struct Vec_and_Mat *vm) {
    size_t k=1;

    for (size_t x=0; x<N; x++) {
        for (size_t y=0; y<N; y++) {
            
            if (vm->field[y][x] != 0) {
                if ((x != 0) && (vm->field[y][x-1] != 0)) {
                    vm->field[y][x] = vm->L[vm->field[y][x-1]-1];
                    vm->S[vm->field[y][x]-1] ++;

                    if ((y!=0) && (vm->field[y-1][x] != 0) && (vm->field[y-1][x] != vm->L[vm->field[y][x-1]-1])) {
                        for (size_t i = 0; i < k; i++)
                            if (vm->L[i] == vm->field[y-1][x]) vm->L[i] = vm->field[y][x];
                        
                        vm->S[vm->L[vm->field[y][x-1]-1]-1] += vm->S[vm->field[y-1][x]-1];
                        vm->S[vm->field[y-1][x]-1] = 0;
                    }
                }

                else if ((y!=0) && (vm->field[y-1][x] != 0)) {
                    vm->field[y][x] = vm->field[y-1][x];
                    vm->S[vm->field[y][x]-1] += 1;
                }           
                        
                else {
                    vm->L = (int *)realloc(vm->L, k * sizeof(int));
                    vm->S = (long long int *)realloc(vm->S, k * sizeof(long long int));
                    vm->L[k-1] = k;
                    vm->field[y][x] = k;
                    vm->S[k-1] = 1;
                    k ++;    
                }
            }
        }
    }

    return k-1;
}