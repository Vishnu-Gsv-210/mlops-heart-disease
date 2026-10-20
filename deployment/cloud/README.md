# Cloud Kubernetes deployment (GKE / EKS / AKS)

The manifests in `deployment/kubernetes/` are portable. Three things change on a
managed cloud cluster: the image registry, the `image:` field, and how you reach
the service. Commands below are illustrative shapes — substitute your own
registry account, region, and project names.

## 1. Build and push the image to a registry

```bash
docker build -t heart-disease-api:latest .

# GKE (Google Artifact Registry)
docker tag heart-disease-api:latest REGION-docker.pkg.dev/PROJECT/REPO/heart-disease-api:1.0.0
docker push REGION-docker.pkg.dev/PROJECT/REPO/heart-disease-api:1.0.0

# EKS (Amazon ECR)
aws ecr create-repository --repository-name heart-disease-api   # once
docker tag heart-disease-api:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/heart-disease-api:1.0.0
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/heart-disease-api:1.0.0

# AKS (Azure Container Registry)
az acr login --name REGISTRYNAME
docker tag heart-disease-api:latest REGISTRYNAME.azurecr.io/heart-disease-api:1.0.0
docker push REGISTRYNAME.azurecr.io/heart-disease-api:1.0.0
```

Prefer a version tag (`:1.0.0`) over `:latest` for cloud deploys so rollouts are
unambiguous.

## 2. Point the Deployment at the pushed image

Edit `deployment/kubernetes/deployment.yaml`:

```yaml
image: REGION-docker.pkg.dev/PROJECT/REPO/heart-disease-api:1.0.0   # GKE example
imagePullPolicy: IfNotPresent
```

For a **private** registry, also add an `imagePullSecrets` block to the pod
spec (create the secret first with `kubectl create secret docker-registry ...`)
— GKE and AKS typically grant pull access automatically when the cluster and
registry are in the same project/subscription; EKS nodes need an ECR pull
policy attached to the node role.

## 3. Apply and verify

```bash
kubectl apply -f deployment/kubernetes/
kubectl rollout status deployment/heart-disease-api -n heart-disease
kubectl get pods,svc -n heart-disease
```

## 4. Reaching the service

- **GKE:** `LoadBalancer` gets an external IP automatically
  (`kubectl get svc -n heart-disease`, watch `EXTERNAL-IP`; needs an LB class
  cluster).
- **EKS:** `LoadBalancer` provisions a Classic/NLB via the AWS Load Balancer
  controller (default on recent EKS; classic ELB otherwise).
- **AKS:** `LoadBalancer` gets a public IP via Azure Load Balancer.
- **Any cluster without an LB controller** (or for TLS/path routing): switch
  `service.yaml` to `type: ClusterIP` and put an **Ingress** in front:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: heart-disease-api
  namespace: heart-disease
  annotations:
    # GKE: default (GCE ingress). EKS: see the AWS Load Balancer controller's
    # group-name annotations. AKS: see the application-routing add-on docs.
    nginx.ingress.kubernetes.io/proxy-body-size: "1m"
spec:
  ingressClassName: nginx            # cluster-dependent; omit on GKE's GCE ingress
  rules:
    - http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: heart-disease-api
                port:
                  number: 80
```

- **Quick test without any ingress:** `kubectl port-forward svc/heart-disease-api 8080:80 -n heart-disease`

## 5. Smoke test

```bash
LB=$(kubectl get svc heart-disease-api -n heart-disease \
     -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
# (EKS: use .hostname instead of .ip)
curl "http://${LB}/health"
curl -X POST "http://${LB}/predict" -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":130,"chol":240,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.2,"slope":2,"ca":0,"thal":3}'
```

## Production notes

- Add authentication, TLS, and rate limiting before exposing the service
  publicly; the API is an educational demo with no built-in auth.
- Persist `/metrics` scraping by deploying Prometheus into the cluster (kube-prometheus-stack) or
  using the provider's managed monitoring: Cloud Monitoring for GKE, CloudWatch
  Container Insights for EKS, Azure Monitor for AKS.
- The model file is baked into the image; for independent model updates, serve it
  from object storage (GCS/S3/Blob) with an init container or sidecar.
