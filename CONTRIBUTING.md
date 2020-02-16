# We are using git flow !
See this cheatsheet for fast info:
https://danielkummer.github.io/git-flow-cheatsheet/index.fr_FR.html

## 1 : Pick or make and issue corresponding to the modification you want to do

## 2 : Start a new feature with the same name as the issue
git flow feature start MYFEATURE 

## 3 : Work on this branch as you wish
If you need another feature : git flow feature pull origin OTHERFEATURE 

## 4 : When your feature is done, finish it (This will do a merge request to dev)
git flow feature finish MYFEATURE

## 5 : If you want to share your feature publish it (This will push to dev)
git flow feature publish MYFEATURE 

## 5 : After testing and buf-fixing, maybe your fix/add will end up on master !