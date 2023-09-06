#!/bin/bash

# ./creer_des_resources.sh
# ./nettoyer_des_resources.sh 


# az login

# Save your login information to variables, and login by CLI
# export login=$(echo "j622amilah@gmail.com")   # <username>
# read -sp "Azure password: " password    # <password>
# az login -u $login -p $password

# ---------------------------------------------

export subscription_id=$(echo "425ecd08-433c-4c16-bcb0-f1ad27233c6d")

az account set -s $subscription_id

# ---------------------------------------------

# Generate a random number that allows you to easily identify which 
# resources belong together. 

# In general, you can keep track of all of your project resources because they are 
# grouped under the same Resource Group. But, I saw tutorials where a random number
# was used in addtion to group project objects.
export val=$(echo "X0")

if [[ $val == "X0" ]]
then 
    let "randomIdentifier=$RANDOM*$RANDOM"
       
else
    let "randomIdentifier=954709494"
fi

# ---------------------------------------------

# Créer Resource Group name

# Identify location name
export location=$(echo "francecentral")
# OU
# export location=$(echo "global") 

# ---------------------------------------------

export val=$(echo "X0")

if [[ $val == "X0" ]]
then 

    export resourceGroupname=$(echo "gdr-name-$randomIdentifier")

    # Run the command below to see a list of location names for your area
    # az account list-locations -o table
    
    # Create a Resource Group
    az group create --name $resourceGroupname --location $location
       
else
    echo "Créer Resource Group : X1"
    export resourceGroupname=$(az group list | jq '.[].name' | tr -d '"' |cut -d $'\n' -f 2)
fi

# Lister des groupes de resources
az group list

# ---------------------------------------------

# Créer Compte de Stockage
export val=$(echo "X0")

if [[ $val == "X0" ]]
then 
    # Storage account name must be between 3 and 24 characters in length and use numbers and lower-case letters only.
    export STORAGE_NAME=$(echo "storagename$randomIdentifier")

    # Create a Storage Account
    az storage account create --name $STORAGE_NAME \
                              --resource-group $resourceGroupname \
                              --location $location \
                              --kind StorageV2 \
                              --sku Standard_LRS \
                              --allow-blob-public-access true
        
    # ---------------------------------------------
    # Obtenir d'information important de compte de stockage
    # ---------------------------------------------
    # Obtenir le connection-string authorization de compte de stockage
    export ids=$(az storage account show -g $resourceGroupname -n $STORAGE_NAME | jq '.id')

    export connectionstring=$(az storage account show-connection-string -g $resourceGroupname -n $STORAGE_NAME | jq '.connectionString')
    # OU
    export connectionstring=$(az storage account show-connection-string -g $resourceGroupname -n $STORAGE_NAME --query connectionString)

    # Lister des comptes de stockage
    az storage account list -g $resourceGroupname

    # Imprimer des clés de stockage
    az storage account keys list -g $resourceGroupname -n $STORAGE_NAME

    # Enregister le premiere clé de compte de stockage 
    export STORAGE_ACCOUNT_KEY=$(az storage account keys list -g $resourceGroupname -n $STORAGE_NAME | jq '.[].value' |cut -d $'\n' -f 1)

    # destination =  https://storagename879395520.file.core.windows.net/
    export destination=$(az storage account show -g $resourceGroupname -n $STORAGE_NAME | jq '.primaryEndpoints.file')
    
    # Verifier que le compte de stockage a access publique
    # https://learn.microsoft.com/en-us/azure/storage/blobs/anonymous-read-access-configure?tabs=azure-cli
    az storage account show \
    --name <storage-account> \
    --resource-group <resource-group> \
    --query allowBlobPublicAccess \
    --output tsv
       
else
    echo "Créer Compte de Stockage : X1"
    export STORAGE_NAME=$(az storage account list -g $resourceGroupname | jq -r '.[].name')
    export ids=$(az storage account show -g $resourceGroupname -n $STORAGE_NAME | jq '.id')
    export connectionstring=$(az storage account show-connection-string -g $resourceGroupname -n $STORAGE_NAME --query connectionString)
    export STORAGE_ACCOUNT_KEY=$(az storage account keys list -g $resourceGroupname -n $STORAGE_NAME | jq '.[].value' |cut -d $'\n' -f 1)
    export destination=$(az storage account show -g $resourceGroupname -n $STORAGE_NAME | jq '.primaryEndpoints.file')
fi

# ---------------------------------------------



# ---------------------------------------------------------
# Créer fichiers partage
# https://learn.microsoft.com/fr-fr/azure/storage/files/storage-how-to-use-files-portal?tabs=azure-cli
# ---------------------------------------------------------
export shareName=$(echo "fichiershare-$randomIdentifier")
export directoryName=$(echo "directory-$randomIdentifier")

export val=$(echo "X0")

if [[ $val == "X0" ]]
then 
    # An Azure file share is a convenient place for cloud applications to write their logs, metrics, and crash dumps. Logs can be written by the application instances via the File REST API, and developers can access them by mounting the file share on their local machine.

    # On a forcé télécharger des fichiers au partage de fichiers
    az storage share-rm create --storage-account $STORAGE_NAME \
        --name $shareName \
        --quota 1024 \
        --enabled-protocols SMB \
        --output none


    # Créer un répertoire nommé myDirectory à la racine de votre partage de fichiers Azure
    az storage directory create \
       --account-name $STORAGE_NAME \
       --share-name $shareName \
       --name $directoryName \
       --output none \
       --connection-string $connectionstring
       
else
    echo "Créer fichiers partage : X1"
fi




# ---------------------------------------------------------
# Charger (Upload) un fichier
# ---------------------------------------------------------
export val=$(echo "X1")

if [[ $val == "X0" ]]
then 
    # Il faut etre dans le meme directory que le fichier 
    cd /home/oem2/Documents/PROGRAMMING/Github_analysis_PROJECTS/Créer_des_questionnaires/Q_Azure

    path_sur_Azure=$(echo "${directoryName}/out0.txt")  # vous mettez le nom de fichier meme s'il n'exist pas sur le serveur
    path_sur_PC_wrt_terminal=$(echo "out0.txt")

    az storage file upload \
        --account-name $STORAGE_NAME \
        --share-name $shareName \
        --source $path_sur_PC_wrt_terminal \
        --path $path_sur_Azure \
        --connection-string $connectionstring

    # az storage file upload --account-name mystorageaccount \
    #                        --account-key $STORAGE_ACCOUNT_KEY \
    #                        --share-name myfileshare \
    #                        --path "myDirectory/index.php" \
    #                        --source "/home/scrapbook/tutorial/php-docs-hello-world/index.php"
    
    az storage file list --account-name $STORAGE_NAME --share-name $shareName --path $path_sur_Azure --output table --connection-string $connectionstring
    
else
    echo "Charger (Upload) un fichier : X1"
fi




# ---------------------------------------------------------
# Charger (Upload) une ensemble des fichiers
# ---------------------------------------------------------
export val=$(echo "X1")

if [[ $val == "X0" ]]
then 
    # Avec le terminal, il faut être dans le fichier ou vous voulez charger
    cd /home/oem2/Documents/PROGRAMMING/Github_analysis_PROJECTS/Créer_des_questionnaires/Q_Azure

    dest_path_sur_Azure=$(echo "${shareName}/${directoryName}")
    dir_path_sur_PC_wrt_terminal=$(echo ".")

    az storage file upload-batch \
        --account-name $STORAGE_NAME \
        --destination $dest_path_sur_Azure \
        --source $dir_path_sur_PC_wrt_terminal \
        --connection-string $connectionstring
    #   --account-key $STORAGE_ACCOUNT_KEY
else
    echo "Ne chargez pas des fichiers au fichier partage : X1"
fi



# ---------------------------------------------------------
# Téléchargement d'un fichier
# ---------------------------------------------------------
# Delete an existing file by the same name as SampleDownload.txt, if it exists, because you've run this example before
# rm -f SampleDownload.txt
export val=$(echo "X1")

if [[ $val == "X0" ]]
then 
    cd /home/oem2/Documents/PROGRAMMING/Github_analysis_PROJECTS/Créer_des_questionnaires
    destpath_sur_PC_wrt_terminal=$(echo ".")

    az storage file download \
        --account-name $STORAGE_NAME \
        --share-name $shareName \
        --path $path_sur_Azure \
        --dest $destpath_sur_PC_wrt_terminal \
        --output none
else
    echo "Ne téléchargez pas un fichier au fichier partage : X1"
fi


# ---------------------------------------------------------
# Supprimer des fichier de - ne fonctionne pas
# ---------------------------------------------------------
export val=$(echo "X1")

if [[ $val == "X0" ]]
then 
    # Avec le terminal, il faut être dans le fichier ou vous voulez charger
    cd /home/oem2/Documents/PROGRAMMING/Github_analysis_PROJECTS/Créer_des_questionnaires/Q_Azure

    dest_path_sur_Azure=$(echo "${shareName}/${directoryName}")
    dir_path_sur_PC_wrt_terminal=$(echo ".")

    az storage file delete-batch \
            --source $dir_path_sur_PC_wrt_terminal \
            --connection-string $connectionstring \
            --pattern '[*.txt]*' 

else
    echo "Ne supprime pas des fichiers au fichier partage : X1"
fi


# ---------------------------------------------------------

# Form Recognizer; OR Cognitive Services et Form Recognizer

# Cognitive Services

az cognitiveservices account list-kinds
# ["AnomalyDetector", "CognitiveServices", "ComputerVision", "ContentModerator", "ConversationalLanguageUnderstanding", "CustomVision.Prediction", "CustomVision.Training", "Face", "FormRecognizer", "HealthInsights", "ImmersiveReader", "Internal.AllInOne", "LUIS", "LUIS.Authoring", "LanguageAuthoring", "MetricsAdvisor", "Personalizer", "QnAMaker.v2", "SpeechServices", "TextAnalytics", "TextTranslation"]


export val=$(echo "X0")

if [[ $val == "X0" ]]
then 

    export COGSERVname=$(echo "cog-servformrecognizer-$randomIdentifier")

    # sku FO = gratuit, 1=standard
    az cognitiveservices account create --name $COGSERVname \
                                        --resource-group $resourceGroupname \
                                        --kind FormRecognizer \
                                        --sku F0 \
                                        --location $location 
                                        --yes

    # https://learn.microsoft.com/fr-fr/cli/azure/cognitiveservices/account/keys?view=azure-cli-latest#az-cognitiveservices-account-keys-list
    az cognitiveservices account keys list --name $COGSERVname -g $resourceGroupname --query key1
    export COGSERV_FormRec_KEY=$(az storage account keys list -g $resourceGroupname -n $STORAGE_NAME | jq '.[].value' |cut -d $'\n' -f 1)
    echo $COGSERV_FormRec_KEY
else
    echo "Ne crée pas cognitive services FormRecognizer : X1"
fi

