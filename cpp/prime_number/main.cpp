#include <vector>
#include <cmath>

#include <iostream>

using namespace std;

#define MAX 0xffffff

int main() {
    vector<int> prime_known;
    prime_known.push_back(2);

    for ( int number = 3; number <= MAX; number+=2 ) {
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
