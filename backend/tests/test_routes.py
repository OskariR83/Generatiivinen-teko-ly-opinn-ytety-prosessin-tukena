# backend/tests/test_routes.py

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_api_status(client):
    response = client.get("/api/test")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "OK"


def test_create_feedback(client):
    response = client.post("/api/feedback?message=Testiviesti")
    
    assert response.status_code == 200
    json_data = response.json()

    assert json_data["message"] == "Testiviesti"
    assert "id" in json_data


def test_list_feedback(client):
    # Lisää yksi merkintä
    client.post("/api/feedback?message=Hei")

    # Testaa listaus
    response = client.get("/api/feedback")
    assert response.status_code == 200

    data = response.json()
    assert len(data) >= 1
    assert "message" in data[0]
