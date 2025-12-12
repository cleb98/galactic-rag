INFO_EXTRACTION_PROMPT = """You are an expert at extracting structured information from unstructured text.
Your task is to analyze the provided text and extract the relevant details into a structured JSON format.
"""

EXTRACTION_EXPLANATION_PROMPT = (
    "Sei un assistente che aiuta a estrarre dati strutturati dal testo di un chunk.\n"
    "Ogni testo descrive un ristorante e il suo menu.\n"
    "I menu descrivono in linguaggio naturale il ristorante, riportando il nome dello Chef, il nome del ristorante, (laddove presente) il pianeta su cui c'è il ristorante e le licenze culinarie che ha lo chef\n"
    "Ogni menu contiene 10 piatti\n"
    "Ogni piatto contiene gli ingredienti usati e le tecniche di preparazione\n"
    "Alcuni menu possiedono anche una descrizione in linguaggio naturale della preparazione\n"
    "Laddove vi siano certi ordini professionali, i menu lo citano\n"
    "Testo del chunk: {document}\n"
    "# COME ESTRARE INFORMAZIONI DA UN TESTO\n"
    "Di seguito ti spiego come estrarre le info di un ristorante da un testo."
    "chef_name: str estrarre il nome dello chef dal testo.\n"
    "restaurant_name: str estrarre il nome del ristorante dal testo.\n"
    "document_summary: str Genera un breve riassunto in italiano con massimo 5 frasi del ristorante descritto nel testo seguente.\n"
    "dish_info: list[DishInfo] Estrai le informazioni sui piatti presenti nel menu e la pagina in cui sono menzionati.\n"
    "Includi il nome del ristorante, dello chef e una descrizione generale.\n"
)
