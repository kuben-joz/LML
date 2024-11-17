resource "google_compute_project_metadata" "default" {
  metadata = {
    ssh-keys = <<EOF
      kuben_joz:ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE7r9veZgsRNXfe7FDTR7XcADWx6CgqtDSKUg1LZHJ0D kuben_joz
    EOF
  }
}
