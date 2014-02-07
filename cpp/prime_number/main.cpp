#include <vector>
#include <cmath>

#include <iostream>

using namespace std;

#define MAX 0xffffffff

int main() {
    vector<int> prime_known;

    for ( int number = 2; number <= MAX; ++number ) {
        int end = ceil( sqrt( number ) );
        bool isPrime = true;
        for ( int divisor_index = 0;
                divisor_index < prime_known.size() && prime_known[divisor_index] <= end;
                ++divisor_index ) {
            if ( number % prime_known[divisor_index] == 0 ) {
                isPrime = false;
                break;
            }
        }
        if ( isPrime ) {
            prime_known.push_back( number );
            cerr << prime_known.size() << "::" << number << "::" << (double)prime_known.size()/(double)number << "%" << endl;
        }
    }

    for ( int i = 0; i < prime_known.size(); ++i ) {
        cout << prime_known[i] << endl;
    }
}
