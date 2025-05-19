# Test Investment Solutions Developer Test - Fintual

## Respuesta pregunta 1
### Justificación método de optimización utilizado
Dado que el problema solicitaba una optimización de portafolios sin un contexto específico, me tomé la libertad de asociarlo a la reforma de las AFP que se avecina durante los próximos años y en la cual Fintual podría jugar un papel importante.

Dentro de los principales aspectos de la reforma se destacan los cambios en la industria de las AFPs que apuntan a fomentar la competencia y mejorar la gestión de los ahorros de los pensionados:
- Cada dos años, se licitará el 10% de los afiliados no pensionados, permitiendo su transferencia a la AFP que ofrezca mejores condiciones, a menos que el afiliado decida permanecer en su AFP actual.
- Se reemplazarán los multifondos por fondos generacionales, asignando a los cotizantes a fondos según su edad, con estrategias de inversión adaptadas al ciclo de vida.

Por ende, es lógico pensar en un método de optimización que se adecue a éstas condiciones. Utilizando la ayuda de la IA, obtuve que las ventajas del modelo Black-Litterman con respecto a otros modelos son:
1) Estabilidad en las asignaciones
    - El modelo evita decisiones extremas al suavizar las estimaciones de retorno. Esto es clave para las AFP, que manejan grandes volúmenes de dinero y deben minimizar riesgos.
2) Incorporación de visiones propias
    - Permite integrar opiniones del equipo gestor sobre el mercado (por ejemplo, inflación, tasas, política). Las AFP pueden adaptar sus decisiones a escenarios específicos del país.
3) Diversificación global coherente
    - Parte desde una cartera de mercado global, promoviendo una diversificación razonable. Las AFP pueden justificar sus inversiones internacionales con base técnica.
4) Compatibilidad con restricciones regulatorias
    - Integra fácilmente límites regulatorios: por tipo de activo, país, emisor o riesgo. Esto facilita cumplir con las normas de la CMF sin afectar la coherencia del portafolio.

### Link al servicio desplegado
https://isd-fintual-test.onrender.com/