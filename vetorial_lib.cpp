#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "comum.h"

void processaVetores(data_t *hmA, data_t *hvB, int nIncognitas) {
    int passo, linha, coluna;
    for (passo = 1; passo < nIncognitas; passo++) {
        int linhaPivo = passo - 1;
        data_t pivo = matriz(hmA, linhaPivo, linhaPivo, nIncognitas);
        for (linha = passo; linha < nIncognitas; linha++) {
            data_t multiplicador =
                matriz(hmA, linha, linhaPivo, nIncognitas) / pivo;
            __m512 vmult = _mm512_set1_ps(multiplicador);
            coluna = linhaPivo;
            for (; coluna <= nIncognitas - 16; coluna += 16) {
                __m512 vlinha =
                    _mm512_loadu_ps(&matriz(hmA, linha, coluna, nIncognitas));
                __m512 vpivo =
                    _mm512_loadu_ps(&matriz(hmA, linhaPivo, coluna, nIncognitas));
                vlinha = _mm512_fnmadd_ps(vpivo, vmult, vlinha);
                _mm512_storeu_ps(&matriz(hmA, linha, coluna, nIncognitas), vlinha);
            }
            /* Tail AVX2 (8 floats) */
            for (; coluna <= nIncognitas - 8; coluna += 8) {
                __m256 vlinha8 =
                    _mm256_loadu_ps(&matriz(hmA, linha, coluna, nIncognitas));
                __m256 vpivo8 =
                    _mm256_loadu_ps(&matriz(hmA, linhaPivo, coluna, nIncognitas));
                __m256 vmult8 = _mm256_set1_ps(multiplicador);
                vlinha8 = _mm256_fnmadd_ps(vpivo8, vmult8, vlinha8);
                _mm256_storeu_ps(&matriz(hmA, linha, coluna, nIncognitas), vlinha8);
            }
            /* Tail escalar */
            for (; coluna < nIncognitas; coluna++) {
                matriz(hmA, linha, coluna, nIncognitas) -=
                    matriz(hmA, linhaPivo, coluna, nIncognitas) * multiplicador;
            }
            matriz(hmA, linha, linhaPivo, nIncognitas) = 0.0f;
            hvB[linha] -= hvB[linhaPivo] * multiplicador;
        }
    }
}