resource "databricks_volume" "mlops_volume" {
  name             = "mi_volume"
  catalog_name     = "mlops_dev"
  schema_name      = "default"   # o el schema que uses
  volume_type      = "MANAGED"  # o EXTERNAL
  comment          = "Volume para MLOps"
}