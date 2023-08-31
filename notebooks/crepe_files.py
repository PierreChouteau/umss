'''This function uses CREPE to individually evaluate pitch for each of the voices commonly used during evaluations.'''

import os

songs = [
    'El_Rossinyol/audio_16kHz/rossinyol_Bajos_207.wav',
	'El_Rossinyol/audio_16kHz/rossinyol_ContraAlt_2-06.wav',
	'El_Rossinyol/audio_16kHz/rossinyol_Soprano_208.wav',
	'El_Rossinyol/audio_16kHz/rossinyol_Tenor2-09.wav',
    'Locus_Iste/audio_16kHz/locus_Bajos_3-02.wav',
	'Locus_Iste/audio_16kHz/locus_ContraAlt_301.wav',
	'Locus_Iste/audio_16kHz/locus_Soprano_310.wav',
	'Locus_Iste/audio_16kHz/locus_tenor3-01-2.wav',
    'Nino_Dios/audio_16kHz/nino_Bajos_404.wav',
	'Nino_Dios/audio_16kHz/nino_ContraAlt_407.wav',
	'Nino_Dios/audio_16kHz/nino_Soprano_405.wav',
	'Nino_Dios/audio_16kHz/nino_tenor4-06-2.wav',
        ]

songs = [
    "Mov1_Cello_Haydn_StringQuartet_op76_n1_b.wav",
    "Mov1_Viola_Haydn_StringQuartet_op76_n1_t.wav",
    "Mov1_Violin1_Haydn_StringQuartet_op76_n1_s.wav",
    "Mov1_Violin2_Haydn_StringQuartet_op76_n1_a.wav",
        ]

for song in songs:
	print("crepe ../Datasets/String/audio_16kHz/{} --step-size 16".format(song))
	os.system("crepe ../Datasets/String/audio_16kHz/{} --step-size 16".format(song))
#the --no-centering flag isn't appropriate!
