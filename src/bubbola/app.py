"""Application logic for Bubbola."""

import logging
from pathlib import Path

from bubbola.config import load_config


class BubbolaApp:
    """
    Main application class for Bubbola.

    This class orchestrates the main functionality of the application,
    including command processing and logging.
    """

    def __init__(self) -> None:
        """Initialize the BubbolaApp."""
        self.config = load_config()

    def run(self, argv: list[str]) -> int:
        """Run the application with given arguments."""
        # Log version information at startup
        from bubbola import __version__

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - Bubbola v%(version)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Add version to log records
        logging.LoggerAdapter(logger, {"version": __version__}).info(
            f"Starting Bubbola v{__version__}"
        )

        if not argv:
            print("Utilizzo: bubbola <comando> [argomenti]")
            print("Per vedere tutti i comandi disponibili: bubbola help")
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
            print("  list      - Lista tutti i flussi di estrazione disponibili")
            print("  extract   - Estrae dati da un batch di immagini")
            return 0
        elif command == "sanitize":
            return self._handle_sanitize_command(argv[1:])
        elif command == "list":
            return self._handle_list_command(argv[1:])
        elif command == "extract":
            return self._handle_extract_command(argv[1:])
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
            print(f"Immagini sanitizzate salvate in: {output_path.as_posix()}")
            return 0
        except Exception as e:
            print(f"Errore durante la sanitizzazione: {e}")
            return 1

    def _handle_list_command(self, args: list[str]) -> int:
        """Handle the list command."""
        if args:
            print("Utilizzo: bubbola list")
            print("  Lista tutti i flussi di estrazione disponibili")
            return 1

        try:
            from bubbola.batch_processor import BatchProcessor

            processor = BatchProcessor()
            flows = processor.list_flows()

            if not flows:
                print("Nessun flusso trovato.")
                return 0

            print("Flussi di estrazione disponibili:")
            for flow in flows:
                print(f"  {flow.name}: {flow.description}")
                print(f"    Modello: {flow.model_name}")
                print(f"    Modello dati: {flow.data_model.__name__}")
                if flow.external_file_options:
                    print("    Opzioni file esterni:")
                    for option, description in flow.external_file_options.items():
                        print(f"      --{option}: {description}")
                print()
            return 0

        except Exception as e:
            print(f"Errore durante il caricamento dei flussi: {e}")
            return 1

    def _handle_extract_command(self, args: list[str]) -> int:
        """Handle the extract command."""
        if not args:
            print("Utilizzo: bubbola extract [opzioni]")
            print("Opzioni:")
            print(
                "  --input <percorso>     - Percorso ai file da elaborare (obbligatorio)"
            )
            print(
                "  --flow <nome_flusso>   - Nome del flusso da utilizzare (obbligatorio)"
            )
            print("  --suppliers-csv <file> - File CSV fornitori (opzionale)")
            print("  --prices-csv <file>    - File CSV prezzi (opzionale)")
            print("  --fattura <file>       - File fattura elettronica (obbligatorio)")
            print(
                "  --yes, -y             - Procede automaticamente senza chiedere conferma"
            )
            print()
            print(
                "Nota: Il comando esegue sempre prima una stima dei costi, poi chiede conferma (a meno che non si usi --yes)."
            )
            print()
            print("Esempi:")
            print("  bubbola extract --input <percorso> --flow <nome_flusso>")
            print(
                "  bubbola extract --input <percorso> --flow delivery_notes --suppliers-csv <file>"
            )
            return 1

        # Validate config
        from bubbola.config import validate_config

        validate_config()

        # Parse arguments manually
        input_path = None
        flow_name = None
        external_files = {}
        auto_confirm = False
        debug = False

        i = 0
        while i < len(args):
            if args[i] == "--input" and i + 1 < len(args):
                input_path = Path(args[i + 1])
                i += 2
            elif args[i] == "--flow" and i + 1 < len(args):
                flow_name = args[i + 1]
                i += 2
            elif args[i] == "--suppliers-csv" and i + 1 < len(args):
                external_files["suppliers_csv"] = Path(args[i + 1])
                i += 2
            elif args[i] == "--prices-csv" and i + 1 < len(args):
                external_files["prices_csv"] = Path(args[i + 1])
                i += 2
            elif args[i] == "--fattura" and i + 1 < len(args):
                external_files["fattura"] = Path(args[i + 1])
                i += 2
            elif args[i] in ["--yes", "-y"]:
                auto_confirm = True
                i += 1
            elif args[i] == "--debug":
                debug = True
                i += 1
            else:
                print(f"Errore: Argomento sconosciuto: {args[i]}")
                return 1

        if not input_path:
            print("Errore: --input è obbligatorio per il comando extract")
            return 1
        if not flow_name:
            print("Errore: --flow è obbligatorio per il comando extract")
            return 1

        try:
            from bubbola.batch_processor import BatchProcessor

            processor = BatchProcessor()

            # Always do a dry run first
            print("=== STIMA COSTI ===")
            dry_run_result = processor.process_batch(
                input_path=input_path,
                flow_name=flow_name,
                dry_run=True,
                external_files=external_files if external_files else None,
            )

            if dry_run_result != 0:
                print("Errore durante la stima dei costi. Interruzione.")
                return dry_run_result

            # Ask for confirmation (unless auto-confirm is enabled)
            if auto_confirm:
                print("\n=== ELABORAZIONE REALE ===")
                return processor.process_batch(
                    input_path=input_path,
                    flow_name=flow_name,
                    dry_run=False,
                    external_files=external_files if external_files else None,
                )
            else:
                print("\n" + "=" * 50)
                try:
                    response = (
                        input("Procedere con l'elaborazione reale? (y/N): ")
                        .strip()
                        .lower()
                    )

                    if response in ["y", "yes", "sì", "si"]:
                        print("\n=== ELABORAZIONE REALE ===")
                        return processor.process_batch(
                            input_path=input_path,
                            flow_name=flow_name,
                            dry_run=False,
                            external_files=external_files if external_files else None,
                        )
                    else:
                        print("Elaborazione annullata dall'utente.")
                        return 0
                except EOFError:
                    print(
                        "Errore: Impossibile leggere l'input. Usa --yes per bypassare la conferma."
                    )
                    return 1

        except Exception as e:
            print(f"Errore durante l'elaborazione: {e}")
            if debug:
                import traceback

                traceback.print_exc()
            return 1
