#include <iostream>

#define MAX 0xfffffffe

int main() {
    bool* is_prime = new bool[MAX];

    for ( unsigned int i = 0; i < MAX; ++i ) {
        is_prime[i] = true;
    }

    for ( unsigned int i = 2; i <= MAX; ++i ) {
        if ( is_prime[i-1] ) {
            for ( unsigned int j = i+i; j <= MAX; j+=i ) {
                is_prime[j-1] = false;
            }
        }
    }

    unsigned int prime = 0;
    unsigned int nonprime = 0;

    for ( unsigned int i = 0; i < MAX; ++i ) {
        if ( is_prime[i] ) {
            ++prime;
            std::cout << i+1 << ":" << (double)prime/(double)nonprime << std::endl;
        } else {
            ++nonprime;
        }
    }

    delete[] is_prime;
}
