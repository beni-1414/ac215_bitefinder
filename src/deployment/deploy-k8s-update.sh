cd deploy_images
pulumi stack select agent
pulumi up --stack agent -y

cd ..
cd deploy_k8s
pulumi stack select agent
pulumi up --stack agent -y
