"""Application logic for Bubbola."""

from pathlib import Path


class BubbolaApp:
    """Main application class for Bubbola."""

    def __init__(self) -> None:
        """Initialize the application."""
        self.config_path = Path.home() / ".bubbola" / "config.json"
        self.config_path.parent.mkdir(exist_ok=True)

    def run(self, argv: list[str]) -> int:
        """Run the application with given arguments."""
        if not argv:
            print("Bubbola - Un'applicazione Python")
            print("Utilizzo: bubbola <comando> [opzioni]")
            return 0

        command = argv[0]

        if command == "version":
            from bubbola import __version__

            print(f"Bubbola version {__version__}")
            return 0
        elif command == "help":
            print("Comandi disponibili:")
            print("  version   - Mostra informazioni sulla versione")
            print("  help      - Mostra questo messaggio di aiuto")
            print(
                "  sanitize  - Converti PDF/immagini in immagini sanitizzate a singola pagina"
            )
            return 0
        elif command == "sanitize":
            return self._handle_sanitize_command(argv[1:])
        else:
            print(f"Comando sconosciuto: {command}")
            print("Usa 'bubbola help' per i comandi disponibili.")
            return 1

    def _handle_sanitize_command(self, args: list[str]) -> int:
        """Handle the sanitize command."""
        if not args:
            print(
                "Utilizzo: bubbola sanitize <input_path> [--output <destination>] [--max-size <pixels>]"
            )
            print(
                "  input_path: Percorso al file PDF, immagine, o cartella contenente i file"
            )
            print(
                "  --output: Cartella di destinazione per le immagini salvate (default: single_pages)"
            )
            print(
                "  --max-size: Dimensione massima dell'edge in pixel per il ridimensionamento delle immagini (opzionale)"
            )
            return 1

        input_path = Path(args[0])
        if not input_path.exists():
            print(f"Errore: Il percorso dell'input non esiste: {input_path}")
            return 1

        # Parse optional arguments
        destination = None
        max_edge_size = None

        i = 1
        while i < len(args):
            if args[i] == "--output" and i + 1 < len(args):
                destination = Path(args[i + 1])
                i += 2
            elif args[i] == "--max-size" and i + 1 < len(args):
                try:
                    max_edge_size = int(args[i + 1])
                    if max_edge_size <= 0:
                        raise ValueError("La dimensione massima deve essere positiva")
                except ValueError as e:
                    print(f"Errore: Valore max-size non valido: {e}")
                    return 1
                i += 2
            else:
                print(f"Errore: Argomento sconosciuto: {args[i]}")
                return 1

        try:
            from bubbola.image_data_loader import save_sanitized_images

            output_path = save_sanitized_images(input_path, destination, max_edge_size)
            print(f"Immagini sanitizzate salvate in: {output_path}")
            return 0
        except Exception as e:
            print(f"Errore durante la sanitizzazione: {e}")
            return 1
