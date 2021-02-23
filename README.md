# Swanky Coin

(WIP title)

## What the hell is this nonsense? 

Our community wants to learn more about NFT tokens and we've decided to create one for fun. 

**What is an NFT?**
NFT stands for Non-fungible token. [Read more on wikipedia](https://en.wikipedia.org/wiki/Non-fungible_token)


## There are a few choices to make: 

1. What blockchain should we issue our NFT on?
2. What do we want the coin to be called? 


# NOTES
1. setup a sandbox algorand node 
resource: https://medium.com/algorand/introducing-sandbox-the-quick-way-to-get-started-on-algorand-8082c2d18854

2. cd into the directory 
3. launch testnet node with: 
```console
$ ./sandbox up testnet
```
4. you can watch the logs with 
```
$ ./sandbox logs
```

5. cool - now how do we issue our NFT?

6. Enter the container with:

```
$ ./sandbox enter algod
```

7. Now lets enter our goal comand on the testnet


8. lets create a new wallet

```
goal wallet new TestSwanky
```
9. follow prompts to create password and save off mneumonic in safe location :-).

10. Let's verify that its been created, we should see the ID!
```goal wallet list``` 

11. Now we need to make an account with that wallet

```
goal account new  <name_of_new_account>
```

see accounts with ```goal account list``` 

12. Now we need to get the mnuemonic for that account 

```
goal account export -a THE-ADDRESS  
```


13. Now we need to get a participation key. [Generate participation key Docs](https://developer.algorand.org/docs/run-a-node/participate/generate_keys/)

I am confused where to get the round firstvalid and last valid should come from .... I just picked these randomly... there is a max of 1000 rounds?

Update: you can check the current round here: https://algoexplorer.io/


```
goal account addpartkey -a <address> --roundFirstValid <latestround>  --roundLastValid <latestround + 1000> 
```

Terminal output to expect: ```Participation key generation successful``` 

We can now Register! 

14. Register Participation key (Online method) [Docs](https://developer.algorand.org/docs/run-a-node/participate/online/)

```
goal account changeonlinestatus --address=RWVB4QGXLR4VNAVTLWNX36QO3BHDXMAJEFEEOGAKS2H4RNXT6IQPUGZFD4 --firstvalid=6002000 --lastvalid=6003000 --online=true --txfile=online.txn
```


15. get some MicroAlgo from the testnet faucet! 

Enter your Public Address

visit: 

16. Verify that you have some MicroAlgo: 
```
goal account list
```

15. Authorize and sign

```
goal clerk sign --infile="online.txn" --outfile="online.stxn"
```

15. Verify 


16. Send some microalgo 

```
goal clerk send --from=RWVB4QGXLR4VNAVTLWNX36QO3BHDXMAJEFEEOGAKS2H4RNXT6IQPUGZFD4 --to=TGNKX4EBIBECBLVX3HH65ZPAMTXZEXSUWP7DPRH2MINRDNT7RKF3JKJXZA --fee=1000 --amount=1000000 --note="test1 send" --out="test1-send.txn"
```

17. sign the transaction 



XX. Create our Asset!

```
goal asset create --asseturl "github.com/Cattleman/swanky-coin" --creator RWVB4QGXLR4VNAVTLWNX36QO3BHDXMAJEFEEOGAKS2H4RNXT6IQPUGZFD4 --decimals 0 --name "TestSwanky" --note "Test flock!" --total 1000 --unitname TSWNK --wallet TestSwankyWallet
```

$ ./goal asset create --asseturl "https://github.com/Cattleman/swanky-coin" --creator < 2aec49fd2195b52daf70e27f617a282b > --decimals 0 --name "TestSwanky" --note "Test flock!" --total 1000 --unitname TSWNK --wallet < TestSwankyWallet >
```



