# chat_test.py
import os
from backend.rag_engine import get_response

def main():
    print("🤖 SkillSync Chat Teszt (Írd be a kérdésed, vagy 'exit' a kilépéshez)")
    print("-" * 50)
    
    while True:
        query = input("\nKérdés: ")
        if query.lower() in ['exit', 'quit', 'kilépés']:
            break
            
        print("🤔 Gondolkozom...")
        answer, sources = get_response(query)
        
        print(f"\n💡 VÁLASZ:\n{answer}")
        print("\n📚 FORRÁSOK:")
        for s in sources:
            print(f"- {s}")

if __name__ == "__main__":
    main()