
#include <iostream>
#include <math.h>

using namespace std;

// constant variables
const int NUM_DIMENSIONS = 10;
const int FUNCTION_VALUE = 255;

/**
 * @brief Some objective function to evaaluate a boolean function
 * 
 * @param boolList A list of boolean values to evaluate
 * @param listLength The length of the list
 * @return int the value produced by the objective function
 */
bool booleanFunction(int* boolList, int listLength){
    int sumOfBool = 0;

    for(int i = 0; i < listLength; i++){
        sumOfBool += boolList[i] * pow(2, listLength-1-i);
    }

    return sumOfBool >= FUNCTION_VALUE;
}

void initHanselChains(int* hanselChains, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    for(int i = 0; i < numChains * numDimensions * numLinks; i++){
        hanselChains[i] = -1;
    }
}

void initHanselChains(int** hanselChains, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    for(int row = 0; row < numChains; row++){
        for(int i = 0; i < numDimensions*numLinks; i++){
            hanselChains[row][i] = -1;
        }
    }
}

void printChain(int* hanselChain, int numDimensions){
    int numLinks = numDimensions + 1;
    for(int i = 0; i < numLinks; i++){
        if(hanselChain[numDimensions * i] != -1){
            for(int j = 0; j < numDimensions; j++){
                cout << hanselChain[numDimensions*i + j];
            }
            if(i != numLinks - 1){cout << " < ";}
        }
    }
    cout << "\n";
}

void printAllChains(int* hanselChains, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    for(int i = 0; i < numChains; i++){
        
        printChain(&hanselChains[i*numLinks*numDimensions], numDimensions);

    }
}

void printAllChains(int** hanselChains, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    for(int i = 0; i < numChains; i++){
        
        printChain(hanselChains[i], numDimensions);

    }
}

void printChainWithoutBreak(int* hanselChain, int numDimensions){
    int numLinks = numDimensions + 1;
    for(int i = 0; i < numLinks; i++){
        if(hanselChain[numDimensions * i] != -1){
            for(int j = 0; j < numDimensions; j++){
                cout << hanselChain[numDimensions*i + j];
            }
            if(i != numLinks - 1){cout << " < ";}
        }
    }
}

void printAllChainsWithResults(int** hanselChains, bool* chainResults, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    for(int i = 0; i < numChains; i++){
        
        printChainWithoutBreak(hanselChains[i], numDimensions);
        cout << " RESULT = " << chainResults[i] << "\n";
    }
}

int getNumActualChains(int* hanselChains, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    int numActualChains = 0;

    for(int i = 0; i < numChains; i++){
        
        // check if the chain is empty
        if(hanselChains[i * numLinks * numDimensions + numLinks * numDimensions - 1] != -1){
            numActualChains++;
        }

    }

    return numActualChains;
}

void shiftChainRightOne(int* hanselChain, int numDimensions){
    int numLinks = numDimensions + 1;

    // shift chain elements right
    for(int i = numLinks*numDimensions-1; i > numDimensions-1; i--){
        hanselChain[i] = hanselChain[i - numDimensions];
    }

    // reset fist numDimensions indices to -1
    for(int i = 0; i < numDimensions; i++){
        hanselChain[i] = -1;
    }
}

bool canShiftRight(int* hanselChain, int numDimensions){
    int numLinks = numDimensions + 1;

    if(hanselChain[numLinks*numDimensions-1] == -1){return true;}
    else{return true;}
}


void shiftChainLeftOne(int* hanselChain, int numDimensions){
    int numLinks = numDimensions + 1;
    // shift chain elements left
    for(int i = 0; i < numLinks*numDimensions-numDimensions; i++){
        hanselChain[i] = hanselChain[i + numDimensions];
    }

    // reset last numDimensions to -1
    for(int i = numLinks * numDimensions - 1; i > numLinks*numDimensions-numDimensions-1; i--){
        hanselChain[i] = -1;
    }
}

bool canShiftLeft(int* hanselChain, int numDimensions){
    if(hanselChain[0] == -1){return true;}
    else{return false;}
}

void cloneAndGrowChains(int* hanselChains, int curNumChains, int numDimensions, int curDimension){
    int numLinks = numDimensions + 1;
    // for each chain
    for(int curChainRow = 0; curChainRow < curNumChains; curChainRow++){
        // get the starting address of the current chain and the new chain
        int newChainRow = curNumChains + curChainRow;

        int* curChain = &hanselChains[curChainRow * numLinks * numDimensions];

        int* newChain = &hanselChains[newChainRow * numLinks * numDimensions];


        // find the first occupied index of the chain
        int firstValueIndex = 0;
        while(curChain[firstValueIndex] == -1) {firstValueIndex++;}
        

        // clone and grow
        for(int j = firstValueIndex; j < (numLinks*numDimensions); j++){

            // if the index is a spot to grow, grow both; old chain gets a 0, new chain grows with a 1
            if(j % numDimensions == firstValueIndex % numDimensions){
                curChain[j-1] = 0;
                newChain[j-1] = 1;
            }
            
            // copy the current value
            newChain[j] = curChain[j];
        }
    }
}

void moveLastVector(int* to, int* from, int numDimensions){
    int numLinks = numDimensions + 1;
    for(int i = numLinks*numDimensions-numDimensions; i < numLinks*numDimensions; i++){
        to[i] = from[i];
        from[i] = -1;
    }
}

void cutAndAddChains(int* hanselChains, int curNumChains, int numDimensions){
    int numLinks = numDimensions + 1;

    // for each original chain
    for(int curChainRow = 0; curChainRow < curNumChains; curChainRow++){

        // get the cur chain and new chain
        int newChainRow = curNumChains + curChainRow;

        int* curChain = &hanselChains[curChainRow * numLinks * numDimensions];

        int* newChain = &hanselChains[newChainRow * numLinks * numDimensions];

        // shift cur chain left
        shiftChainLeftOne(curChain, numDimensions);

        // move last vector from new chain to curChain
        moveLastVector(curChain, newChain, numDimensions);

        // shift new chain right
        shiftChainRightOne(newChain, numDimensions);
    }
}

void storeActualChains(int** actualChains, int* roughChains, int numRoughChains, int numDimensions){
    int numLinks = numDimensions + 1;

    int actualChainRow = 0;
    for(int roughChainRow = 0; roughChainRow < numRoughChains; roughChainRow++){
        
        // check if the chain is empty
        if(roughChains[roughChainRow * numLinks * numDimensions + numLinks * numDimensions - 1] != -1){
            
            // iterate through the chain and store the values
            for(int j = 0; j < numLinks * numDimensions; j++){
                actualChains[actualChainRow][j] = roughChains[roughChainRow * numLinks * numDimensions + j];
            }

            actualChainRow++;
        }

    }
    
}

int getNumLinksInChain(int* chain, int numDimensions){
    int numLinks = numDimensions + 1;

    int numLinksInChain = 0;

    for(int i = 0; i < numLinks; i++){
        if(chain[i * numDimensions - 1] != -1){
            numLinksInChain++;
        }
    }

    return numLinksInChain;
}

void duplicateChain(int* newChainLocation, int* chainToDuplicate, int numDimensions){
    int numLinks = numDimensions + 1;

    for(int i = 0; i < numDimensions * numLinks; i++){
        newChainLocation[i] = chainToDuplicate[i];
    }
}
void swapAndShift(int** chains, int formerRowIndex, int tempRowIndex, int numDimensions){

    int numLinks = numDimensions + 1;

    int* duplicateTempChain = (int*)malloc(sizeof(int) * numLinks * numDimensions);

    duplicateChain(duplicateTempChain, chains[tempRowIndex], numDimensions);


    for(int i = tempRowIndex; i > formerRowIndex; i--){
        chains[i] = chains[i-1];
    }


    // overwrite chains
    chains[formerRowIndex] = duplicateTempChain;

}

/**
 * @brief Sorts chains using a basic n^2 sorting method
 * 
 * @param chains A complete list of chains to be sorted
 * @param numChains The number of chains in chains
 * @param numDimensions The number of dimensions represented in chains
 */
void sortChains(int** chains, int numChains, int numDimensions){
    int numLinks = numDimensions + 1;

    // use a basic stable sorting method in n^2 time
    for(int chainRow = 0; chainRow < numChains; chainRow++){
        int numLinksInCurChain = getNumLinksInChain(chains[chainRow], numDimensions);

        for(int tempRow = chainRow; tempRow < numChains; tempRow++){
            int numLinksInTempChain = getNumLinksInChain(chains[tempRow], numDimensions);

            // if temp chain is less than cur chain, swap addresses and move all address forward
            if(numLinksInTempChain < numLinksInCurChain){
                // swap and shift
                swapAndShift(chains, chainRow, tempRow, numDimensions);
            }
        }
    }
}

int* getFirstLink(int* chain, int numDimensions){
    int numLinks = numDimensions + 1;
    for(int i = 0; i < numLinks; i++){
        if(chain[i * numDimensions] != -1){
            return &chain[i * numDimensions];
        }
    }
}

int getLinkValue(int* link, int linkLength){
    int linkSum = 0;

    for(int i = 0; i < linkLength; i++){
        linkSum += link[i] * pow(2, linkLength - i - 1);
    }

    return linkSum;
}

int restoreFunctionValue(int** chains, bool* resultsPerChain, int numChains, int numDimensions){
    int functionValue = INT_MAX;

    for(int chainRow = 0; chainRow < numChains; chainRow++){
        int* firstLink = getFirstLink(chains[chainRow], numDimensions);

        int firstLinkValue = getLinkValue(firstLink, numDimensions);

        if(resultsPerChain[chainRow] == true && firstLinkValue < functionValue){
            functionValue = firstLinkValue;
        }
    }

    return functionValue;

}

int getBestValueInChain(int* chain, int numDimensions){
    int numLinks = numDimensions + 1;

    int smallestVectorValue = INT_MAX;
    // get each link, and evaluate against booleanFunction, storing the smallest vector value
    int* link = &chain[0];

    for(int curLink = 0; curLink < numLinks; curLink++){
        if(link[0] != -1){

            if(booleanFunction(link, numDimensions)){
                int linkValue = getLinkValue(link, numDimensions);

                if(linkValue < smallestVectorValue){
                    smallestVectorValue = linkValue;
                }
            }
        }

        link = &link[numDimensions];
    }

    return smallestVectorValue;
}

int main(){
    // allocate memory for hansel chains
    // we know the maximum length of a hansel chain to be equal to the number of dimensions
    // we also know that the maximum number of hansel chains is equal to 2^NUM_DIMENSIONS
    int numChains = pow(2, NUM_DIMENSIONS-1);
    int* hanselChains = (int*)malloc(sizeof(int) * numChains * (NUM_DIMENSIONS+1) * NUM_DIMENSIONS);
    initHanselChains(hanselChains, numChains, NUM_DIMENSIONS);

    // initialize the first hansel chain for a single dimension
    hanselChains[NUM_DIMENSIONS*(NUM_DIMENSIONS+1) - 1 - NUM_DIMENSIONS] = 0;
    hanselChains[NUM_DIMENSIONS*(NUM_DIMENSIONS+1) - 1] = 1;


    
    // build chains
    for(int i = 1; i < NUM_DIMENSIONS; i++){
        // clone existing chains
        int curNumChains = pow(2, i-1);
        cloneAndGrowChains(hanselChains, curNumChains, NUM_DIMENSIONS, i+1);
        cutAndAddChains(hanselChains, curNumChains, NUM_DIMENSIONS);
        
    }

    int numActualChains = getNumActualChains(hanselChains, numChains, NUM_DIMENSIONS);

    // allocate space for and store all real chains
    int** actualHanselChains = (int**)malloc(sizeof(int*)* numActualChains);
    for(int i = 0; i < numActualChains; i++){
        actualHanselChains[i] = (int*)malloc(sizeof(int) * NUM_DIMENSIONS * (NUM_DIMENSIONS + 1));
    }

    initHanselChains(actualHanselChains, numActualChains, NUM_DIMENSIONS);

    storeActualChains(actualHanselChains, hanselChains, numChains, NUM_DIMENSIONS);

    // sort chains by length
    sortChains(actualHanselChains, numActualChains, NUM_DIMENSIONS);

    // evalute chains to identify positive cases
    bool functionIdentified = false;

    bool* resultsPerChain = (bool*)malloc(sizeof(bool) * numActualChains);

    // iterate through the fist indeces of each chain and evaluate the result of boolFunction
    for(int i = 0; i < numActualChains; i++){
        int* curChainLink = getFirstLink(actualHanselChains[i], NUM_DIMENSIONS);

        // evaluate curChainLink
        resultsPerChain[i] = booleanFunction(curChainLink, NUM_DIMENSIONS);
    }

    // print chains and their results
    cout << "Chains With Results:\n";
    printAllChainsWithResults(actualHanselChains, resultsPerChain, numActualChains, NUM_DIMENSIONS);

    int restoredFunctionValue = restoreFunctionValue(actualHanselChains, resultsPerChain, numActualChains, NUM_DIMENSIONS);

    // for the chain with a true value following a 1, iterate through the chain to locate the changing point
    for(int i = 0; i < numActualChains; i++){
        if(resultsPerChain[i] == false){
            int bestValueInChain = getBestValueInChain(actualHanselChains[i], NUM_DIMENSIONS);

            if(bestValueInChain < restoredFunctionValue){
                restoredFunctionValue = bestValueInChain;
            }
        }
    }

    cout << "RESTORED FUNCTION VALUE = " << restoredFunctionValue << "\n";
    
    free(hanselChains);

    for(int i = 0; i < numActualChains; i++){
        free(actualHanselChains[i]);
    }

    free (resultsPerChain);
}