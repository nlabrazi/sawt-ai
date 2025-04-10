curl -k -X PUT -u admin:MonSuperMotDePasse123! "https://localhost:9200/_cluster/settings" -H "Content-Type: application/json" -d '
{
  "persistent": {
    "knn.plugin.enabled": true
  }
}'

Activate knn
