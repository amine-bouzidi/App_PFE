import { NextRequest } from "next/server";
import { runScraper, runAnalysis, runAdvancedAnalysis } from "@/services/pythonRunner";
import { prisma } from "@/lib/db";
import fs from "fs";

// Force le rendu dynamique de cette route (pas de mise en cache statique)
export const dynamic = "force-dynamic";

/**
 * Endpoint POST principal pour démarrer le processus de scraping et d'analyse IA.
 * Utilise les Server-Sent Events (SSE) pour diffuser en temps réel les journaux d'exécution (logs) à l'interface utilisateur.
 */
export async function POST(request: NextRequest) {
  try {
    // Récupération des paramètres envoyés par le client
    const { projectId, keyword, topic, sources, limit, mock } = await request.json();

    // Validation rudimentaire des entrées requises
    if (!projectId || !keyword || !sources) {
      return new Response(JSON.stringify({ error: "Missing required parameters" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Création initiale de l'enregistrement Dataset en base de données avec l'état 'RUNNING'
    const dataset = await prisma.dataset.create({
      data: {
        name: `Scrape - ${keyword} (${new Date().toLocaleDateString()})`,
        status: "RUNNING",
        projectId: projectId,
        logs: "",
      },
    });

    const encoder = new TextEncoder();
    let accumulatedLogs = "";

    // Instanciation d'un ReadableStream pour le streaming des logs SSE
    const stream = new ReadableStream({
      async start(controller) {
        // Helper pour envoyer des messages structurés au client SSE
        const sendUpdate = (data: any) => {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
        };

        sendUpdate({ status: "running", message: "Starting scrapers...", datasetId: dataset.id });

        // 1. Démarrage de l'exécution des scripts de scraping Python en arrière-plan
        runScraper(
          { keyword, topic, sources, limit: Number(limit) || 10, mock: !!mock },
          async (logLine) => {
            // Callback invoqué à chaque ligne de log reçue sur stdout du processus de scraping
            accumulatedLogs += logLine;
            sendUpdate({ status: "scraping", log: logLine });
          },
          async (datasetPath) => {
            // Callback de fin du processus de scraping
            if (!datasetPath) {
              // Si aucun chemin de base de données n'est renvoyé, le scraping a échoué
              sendUpdate({ status: "failed", message: "Scraping failed to produce a dataset." });
              await prisma.dataset.update({
                where: { id: dataset.id },
                data: { status: "FAILED", logs: accumulatedLogs },
              });
              controller.close();
              return;
            }

            try {
              sendUpdate({ status: "parsing", message: "Parsing scraped data into database..." });
              
              // Lecture du fichier de résultats JSON produit par le script Python
              const rawData = fs.readFileSync(datasetPath, "utf-8");
              const items = JSON.parse(rawData);

              // 2. Insertion des publications récupérées dans la base de données SQLite
              for (const item of items) {
                await prisma.datasetItem.create({
                  data: {
                    title: item.title,
                    content: item.content,
                    source: item.source,
                    author: item.author,
                    timestamp: new Date(item.timestamp),
                    url: item.url,
                    sentiment: "NEUTRAL", // Sentiment initial neutre avant l'analyse NLP
                    datasetId: dataset.id,
                  },
                });
              }

              // Mise à jour du compteur de publications collectées
              await prisma.dataset.update({
                where: { id: dataset.id },
                data: { itemCount: items.length, logs: accumulatedLogs },
              });

              sendUpdate({ status: "analyzing", message: "Scraping finished. Launching AI NLP engine..." });

              // 3. Lancement du script d'analyse NLP d'intelligence artificielle
              const analysisOutput = await runAnalysis(datasetPath, (aiLog) => {
                accumulatedLogs += aiLog;
                sendUpdate({ status: "analyzing_logs", log: aiLog });
              });

              sendUpdate({ status: "advanced_analyzing", message: "Running advanced Python analysis modules..." });

              let advancedAnalysisOutput: any = null;
              try {
                advancedAnalysisOutput = await runAdvancedAnalysis(datasetPath, (advancedLog) => {
                  accumulatedLogs += advancedLog;
                  sendUpdate({ status: "advanced_logs", log: advancedLog });
                });
              } catch (advancedErr: any) {
                accumulatedLogs += `\nAdvanced analysis failed: ${advancedErr.message}`;
                sendUpdate({
                  status: "advanced_warning",
                  message: `Advanced analysis partially failed: ${advancedErr.message}`,
                });
              }

              sendUpdate({ status: "saving", message: "Saving analysis reports..." });

              // 4. Enregistrement des statistiques d'analyse IA en base de données
              const analysis = await prisma.analysis.create({
                data: {
                  status: "COMPLETED",
                  sentimentDistribution: JSON.stringify(analysisOutput.sentimentDistribution),
                  topicClusters: JSON.stringify(analysisOutput.topicClusters),
                  temporalData: JSON.stringify(analysisOutput.temporalData),
                  summary: analysisOutput.report.summary,
                  projectId: projectId,
                  datasetId: dataset.id,
                },
              });

              // 5. Génération du rapport de synthèse stratégique (Executive Report)
              const report = await prisma.report.create({
                data: {
                  title: `Executive Report: ${keyword} - ${topic}`,
                  summary: analysisOutput.report.summary,
                  keyInsights: JSON.stringify(analysisOutput.report.keyInsights),
                  conclusions: JSON.stringify(analysisOutput.report.conclusions),
                  projectId: projectId,
                  analysisId: analysis.id,
                },
              });

              // 6. Mise à jour rétroactive des sentiments de chaque publication
              // Les données étiquetées renvoyées par analyze.py écrasent la valeur 'NEUTRAL' initiale
              const updatedRawData = fs.readFileSync(datasetPath, "utf-8");
              const updatedItems = JSON.parse(updatedRawData);
              for (const uItem of updatedItems) {
                await prisma.datasetItem.updateMany({
                  where: {
                    datasetId: dataset.id,
                    title: uItem.title,
                  },
                  data: {
                    sentiment: uItem.sentiment,
                  },
                });
              }

              // Finalisation du Dataset en base de données avec l'état 'COMPLETED'
              await prisma.dataset.update({
                where: { id: dataset.id },
                data: {
                  status: "COMPLETED",
                  logs: accumulatedLogs + (
                    advancedAnalysisOutput
                      ? `\nAdvanced analysis outputs: ${advancedAnalysisOutput.output_dir}`
                      : ""
                  ),
                },
              });

              // Envoi de la charge utile de réussite globale avec tous les IDs générés
              sendUpdate({
                status: "completed",
                message: "InsightFlow AI pipelines finished successfully!",
                datasetId: dataset.id,
                analysisId: analysis.id,
                reportId: report.id,
                advancedOutputDir: advancedAnalysisOutput?.output_dir || null,
              });

            } catch (err: any) {
              console.error("Pipeline post-processing error:", err);
              sendUpdate({ status: "failed", message: `Error processing data: ${err.message}` });
              await prisma.dataset.update({
                where: { id: dataset.id },
                data: { status: "FAILED", logs: accumulatedLogs + `\nException: ${err.message}` },
              });
            } finally {
              controller.close();
            }
          }
        );
      },
    });

    // Retourne la réponse en text/event-stream pour maintenir la connexion ouverte
    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (error: any) {
    console.error("Scraper route error:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
