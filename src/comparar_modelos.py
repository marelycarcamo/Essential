
import time

from consumir_deepseek import armar_peticion, enviar_peticion


if __name__ == "__main__":
    pregunta = "¿Qué es la inteligencia artificial?"

    url, headers, body = armar_peticion(pregunta)

    inicio = time.perf_counter()
    datos = enviar_peticion(url, headers, body)
    fin = time.perf_counter()

    latencia = fin - inicio
    uso = datos.get("usage", {})

    print(f"Proveedor: DeepSeek")
    print(f"Modelo solicitado: {body['model']}")
    print(f"Modelo informado: {datos.get('model', 'No informado')}")
    print(f"Latencia: {latencia:.2f} segundos")
    print(f"Tokens de entrada: {uso.get('prompt_tokens', 'No informado')}")
    print(f"Tokens de salida: {uso.get('completion_tokens', 'No informado')}")
    print(f"Tokens totales: {uso.get('total_tokens', 'No informado')}")
