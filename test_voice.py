from eva.adapters.voice import KokoroSpeaker, WhisperTranscriber, Microphone

print("1. Carregando Kokoro na CPU (deve demorar um pouco mais, mas não dar OOM)...")
speaker = KokoroSpeaker(language="p")
print("   -> Kokoro carregado! Sintetizando fala...")
# Descomente para testar a fala (pode assustar)
# speaker.speak("Olá, eu sou a Eva. Minhas dependências de voz estão prontas.")

print("2. Carregando Whisper na CPU...")
whisper = WhisperTranscriber(model_name="tiny") # tiny para ser rápido
print("   -> Whisper carregado!")

print("\nTudo certo! Os modelos estão na CPU e não vão brigar com o Qwen pela sua VRAM.")
