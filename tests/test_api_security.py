"""Security tests: path traversal, auth, CORS, upload limits."""



class TestAuthentication:
    """Verify API key authentication is enforced."""

    def test_auth_required_no_key(self, client):
        """Verify 401 without API key."""
        resp = client.get("/api/runs")
        assert resp.status_code == 401

    def test_auth_required_wrong_key(self, client):
        """Verify 403 with invalid API key."""
        resp = client.get("/api/runs", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_auth_success(self, client, auth_headers):
        """Verify 200 with correct API key."""
        resp = client.get("/api/runs", headers=auth_headers)
        assert resp.status_code == 200

    def test_health_no_auth(self, client):
        """Health endpoint should not require auth."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPathTraversal:
    """Verify path traversal attacks are blocked."""

    def test_chart_traversal_batch_id(self, client, auth_headers):
        """Verify ../../../etc/passwd in batch_id is rejected."""
        resp = client.get("/api/runs/../../../etc/passwd/charts", headers=auth_headers)
        # Should either 404 or 403, not serve the file
        assert resp.status_code in (403, 404, 422)

    def test_chart_traversal_chart_name(self, client, auth_headers):
        """Verify ../../../etc/passwd in chart_name is rejected."""
        resp = client.get(
            "/api/runs/test_batch/charts/../../etc/passwd",
            headers=auth_headers,
        )
        assert resp.status_code in (403, 404)

    def test_chart_name_with_dotdot(self, client, auth_headers):
        """Verify .. in chart name is rejected."""
        resp = client.get(
            "/api/runs/test_batch/charts/..%2F..%2Fetc%2Fpasswd",
            headers=auth_headers,
        )
        assert resp.status_code in (403, 404)


class TestUploadSecurity:
    """Verify upload validations."""

    def test_upload_non_csv_rejected(self, client, auth_headers):
        """Verify non-CSV files are rejected."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("test.txt", b"not a csv", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_size_limit(self, client, auth_headers):
        """Verify 413 on oversized upload."""
        # Create a file larger than MAX_UPLOAD_SIZE (10MB)
        large_content = b"a,b,c\n" + b"1,2,3\n" * (2 * 1024 * 1024)  # ~12MB
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("big.csv", large_content, "text/csv")},
        )
        assert resp.status_code == 413

    def test_upload_invalid_csv_rejected(self, client, auth_headers):
        """Verify invalid CSV content is rejected."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("test.csv", b"\x00\x01\x02binary", "text/csv")},
        )
        assert resp.status_code == 400

    def test_upload_valid_csv_accepted(self, client, auth_headers, sample_csv_content):
        """Verify valid CSV upload is accepted."""
        resp = client.post(
            "/api/ingest",
            headers=auth_headers,
            files={"file": ("test_data.csv", sample_csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_id" in data
        assert data["status"] == "queued"
