import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "../App";

// Mockataan fetch, jotta backend-kutsut eivät oikeasti lähde testissä.
// Palautetaan aina onnistunut tyhjä vastaus.
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
  })
);

describe("Feedback modal", () => {
  test("modal opens and closes", () => {
    // Renderöidään koko App
    render(<App />);

    // Avataan palaute-modali klikkaamalla palaute-nappia
    fireEvent.click(screen.getByText(/Palaute/i));

    // Tarkistetaan että modalin otsikko näkyy — modali on siis avautunut
    expect(screen.getByText(/Anna palautetta/i)).toBeInTheDocument();

    // Suljetaan modali klikkaamalla X-nappia
    fireEvent.click(screen.getByText("✖"));

    // Modalin ei pitäisi enää näkyä
    expect(screen.queryByText(/Anna palautetta/i)).not.toBeInTheDocument();
  });

  test("sending feedback triggers toast", async () => {
    render(<App />);

    // Avataan palaute-modali
    fireEvent.click(screen.getByText(/Palaute/i));

    // Kirjoitetaan palautteeseen tekstiä
    fireEvent.change(
      screen.getByPlaceholderText(/Kirjoita palautteesi/i),
      { target: { value: "Hyvä botti!" } }
    );

    // Lähetetään palaute klikkaamalla Lähetä-nappia
    fireEvent.click(screen.getByText("Lähetä"));

    // Odotetaan että toast-viesti ilmestyy (koska setTimeout ja state-päivitys)
    const toast = await screen.findByText(/Kiitos palautteestasi/i);
    // Varmistetaan että toast näkyy näytöllä
    expect(toast).toBeInTheDocument();
  });
});
