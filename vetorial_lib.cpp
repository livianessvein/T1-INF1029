/**
 * @file vetorial_lib.cpp
 * @brief Implementação vetorizada (AVX/FMA) da eliminação de Gauss para sistemas lineares Ax = b.
 *
 * Este módulo implementa a função processaVetores utilizando instruções vetoriais
 * AVX (Advanced Vector Extensions) e FMA (Fused Multiply-Add) da biblioteca Intel
 * Intrinsics, processando 8 floats por ciclo para maximizar o throughput.
 *
 * Trabalho 1 – 2026.1
 * Livian Essvein 2211667 e Breno Lois 2210780
 * Introdução à Arquitetura de Computadores
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>   /* AVX / FMA intrinsics */
#include "comum.h"

/**
 * @brief Zera todos os elementos abaixo da diagonal principal de A usando
 *        eliminação de Gauss, com vetorização AVX/FMA.
 *
 * Para cada passo k (coluna pivô), e para cada linha abaixo do pivô:
 *   1. Calcula o multiplicador  m = A[linha][k-1] / A[k-1][k-1]
 *   2. Atualiza cada coluna c >= k-1:
 *        A[linha][c] -= A[k-1][c] * m          (operação FMA: a - b*c)
 *   3. Atualiza o vetor b:
 *        b[linha]    -= b[k-1]    * m
 *
 * O laço interno sobre colunas processa 8 floats por iteração (__m256).
 * O restante (tail) é tratado escalarmente para garantir correção quando
 * nIncognitas não é múltiplo de 8.
 *
 * @param hmA        Ponteiro para a matriz A em row-major (nIncognitas x nIncognitas floats).
 * @param hvB        Ponteiro para o vetor b (nIncognitas floats).
 * @param nIncognitas Número de incógnitas / tamanho do sistema.
 */
void processaVetores(data_t *hmA, data_t *hvB, int nIncognitas) {

    /* passo: índice da coluna pivô (começa em 1, pois a coluna 0 já é o 1º pivô) */
    for (int passo = 1; passo < nIncognitas; passo++) {

        /* ponteiro para o início da linha pivô (linha passo-1) */
        data_t *linhaPivo = &matriz(hmA, passo - 1, 0, nIncognitas);
        data_t  bPivo     = hvB[passo - 1];

        for (int linha = passo; linha < nIncognitas; linha++) {

            /* ----------------------------------------------------------------
             * Calcula o multiplicador m = A[linha][passo-1] / A[passo-1][passo-1]
             * e o espalha em todos os 8 raias de um registrador AVX.
             * ---------------------------------------------------------------- */
            data_t multiplicador =
                matriz(hmA, linha, passo - 1, nIncognitas) /
                matriz(hmA, passo - 1, passo - 1, nIncognitas);

            /* _mm256_set1_ps replica o escalar nos 8 floats do registrador */
            __m256 vMult = _mm256_set1_ps(multiplicador);

            data_t *linhaAtual = &matriz(hmA, linha, 0, nIncognitas);

            /* ----------------------------------------------------------------
             * Laço vetorial: processa 8 colunas por iteração.
             * Começa em passo-1 para incluir a coluna que será zerada e
             * garantir que a aritmética FMA seja consistente com a versão escalar.
             * ---------------------------------------------------------------- */
            int coluna = passo - 1;
            int limite = nIncognitas - 8; /* última posição segura para carregar 8 floats */

            for (; coluna <= limite; coluna += 8) {
                /* Carrega 8 floats da linha pivô e da linha atual */
                __m256 vPivo  = _mm256_loadu_ps(&linhaPivo[coluna]);
                __m256 vAtual = _mm256_loadu_ps(&linhaAtual[coluna]);

                /*
                 * FMA: vAtual = vAtual - vPivo * vMult
                 * _mm256_fnmadd_ps(a, b, c) = c - a*b
                 *   → linhaAtual[c] -= linhaPivo[c] * multiplicador
                 */
                vAtual = _mm256_fnmadd_ps(vPivo, vMult, vAtual);

                /* Escreve o resultado de volta */
                _mm256_storeu_ps(&linhaAtual[coluna], vAtual);
            }

            /* ----------------------------------------------------------------
             * Tail escalar: colunas restantes (quando nIncognitas % 8 != 0)
             * ---------------------------------------------------------------- */
            for (; coluna < nIncognitas; coluna++) {
                linhaAtual[coluna] -= linhaPivo[coluna] * multiplicador;
            }

            /* Atualiza o vetor b escalarmente (um único elemento por linha) */
            hvB[linha] -= bPivo * multiplicador;
        }
    }
}
