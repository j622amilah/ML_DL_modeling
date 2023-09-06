#!/bin/bash

# export resourceGroupname=$(echo "gdr-name-879395520")
# On a fait (.[]) parce que le JSON etait dans []
# Supprimer des guillmets avec (tr -d '"')
export resourceGroupname=$(az group list | jq '.[].name' | tr -d '"')

# export STORAGE_NAME=$(az storage account list -g $resourceGroupname | jq -r '.[].name')

# Supprimer toutes des Resource Groupe
for i in $resourceGroupname
do
   az group delete --name $i --no-wait --yes
done
